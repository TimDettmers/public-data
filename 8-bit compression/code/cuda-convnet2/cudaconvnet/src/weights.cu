/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <map>
#include <algorithm>
#include "../include/weights.cuh"
#include "../include/lr.cuh"
#include "../include/worker.cuh"

using namespace std;

/* ========================
 * IWeightReducer
 * ========================
 */
int IWeightReducer::getDeviceID() {
    return _replicas[_tgtReplicaID]->getDeviceID();
}

IWeightReducer::IWeightReducer(std::map<int,Weights*>& replicas, int tgtReplicaID) : _replicas(replicas), _tgtReplicaID(tgtReplicaID) {
}

IWeightReducer::~IWeightReducer() {
}

IWeightReducer& IWeightReducer::make(std::map<int,Weights*>& replicas, int tgtReplicaID) {
    if (replicas.size() == 8) {
        return *new ParallelWeightReducer(replicas, tgtReplicaID);
    }
    return *new SequentialWeightReducer(replicas, tgtReplicaID);
}

/* ========================
 * SequentialWeightReducer
 * ========================
 */
SequentialWeightReducer::SequentialWeightReducer(std::map<int,Weights*>& replicas, int tgtReplicaID) : IWeightReducer(replicas, tgtReplicaID) {
    _sb = new StreamBroadcast();
}

SequentialWeightReducer::~SequentialWeightReducer() {
    delete _sb;
}

void SequentialWeightReducer::reduce(std::map<int, NVMatrix*> gradShards, float gradScale, bool toInc) {
    std::map<int, NVMatrix*> mats; // device id -> grad
    mats[getDeviceID()] = toInc ? &_replicas[_tgtReplicaID]->getInc() : &_replicas[_tgtReplicaID]->getGrad();
    for (int i = 0, r = _tgtReplicaID; i < _replicas.size(); ++i, r = (r + 1) % _replicas.size()) {
        if (r != _tgtReplicaID) {
            mats[_replicas[r]->getDeviceID()] = gradShards[r];
            _sb->transfer(mats, _replicas[r]->getDeviceID(), 1, gradScale);
            mats.erase(_replicas[r]->getDeviceID());
        }
    }
}

/* ========================
 * ParallelWeightReducer
 * ========================
 */
ParallelWeightReducer::ParallelWeightReducer(std::map<int,Weights*>& replicas, int tgtReplicaID) : IWeightReducer(replicas, tgtReplicaID) {
    _reducer = &(new EightGPUReducer1(getDeviceID()))->construct();
}

ParallelWeightReducer::~ParallelWeightReducer() {
    delete _reducer;
}

void ParallelWeightReducer::reduce(std::map<int, NVMatrix*> gradShards, float gradScale, bool toInc) {
    std::map<int, NVMatrix*> mats; // device id -> grad
    mats[getDeviceID()] = toInc ? &_replicas[_tgtReplicaID]->getInc() : &_replicas[_tgtReplicaID]->getGrad();
    for (std::map<int,Weights*>::const_iterator it = _replicas.begin(); it != _replicas.end(); ++it) {
        if (it->first != _tgtReplicaID) {
            mats[it->second->getDeviceID()] = gradShards[it->first];
        }
    }
    _reducer->reduce(mats, gradScale, 1);
}

// weights has pointer to layer, layer pointer to thread
// thread has sync (copy) object for every other thread
// weights uses copy object to sum grad contributions into inc matrix slice (phase 1)
// weights broadcasts inc matrix slice to other inc matrix replicas (phase 2)

NVMatrix& Weights::operator*() const {
    return getW();
}

/*
 * TODO: get rid of this constructor duplication.
 */
Weights::Weights(Weights& srcWeights, ParameterSchedule& lrs, Layer& parent) {
    init(srcWeights.getCPUW(), srcWeights.getCPUWInc(), lrs, parent, 0, 0, srcWeights.getMom(), srcWeights.isUseGrad(), false);
    _srcWeights = &srcWeights;
}

Weights::Weights(Matrix& hWeights, Matrix& hWeightsInc, ParameterSchedule& lrs, Layer& parent, float wc,
                 float wball, float mom, bool useGrad) {
    init(hWeights, hWeightsInc, lrs, parent, wc, wball, mom, useGrad, true);
}

void Weights::init(Matrix& hWeights, Matrix& hWeightsInc, ParameterSchedule& lrs, Layer& parent, float wc,
              float wball, float mom, bool useGrad, bool cleanup) {
    _srcWeights = NULL;
    _hWeights = &hWeights;
    _hWeightsInc = &hWeightsInc;
    _numUpdates = 0;
    _lrs = &lrs;
    _parent = &parent;
    _wc = wc;
    _wball = wball;
    _mom = mom;
    _useGrad = useGrad;
    _onGPU = false;
    _weights = NULL;
    _weightsInc = NULL;
    _weightsGrad = NULL;
    _cleanup = cleanup;
    _reducer = NULL;
    _broadcaster = NULL;

    _buffer8bit = NULL;
    _abs_buffer = NULL;


	float data[126] =  {0.000003, 0.000007, 0.000019, 0.000036, 0.000059, 0.000086, 0.000144, 0.000231, 0.000319, 0.000406, 0.000519, 0.000656, 0.000794, 0.000931, 0.001219, 0.001656, 0.002094,
	0.002531, 0.002969, 0.003406, 0.003844, 0.004281, 0.004844, 0.005531, 0.006219, 0.006906, 0.007594, 0.008281, 0.008969, 0.009656, 0.011094, 0.013281, 0.015469, 0.017656, 0.019844, 0.022031, 0.024219, 0.026406, 0.028594, 0.030781, 0.032969, 0.035156, 0.037344, 0.039531, 0.041719, 0.043906, 0.046719, 0.050156, 0.053594, 0.057031, 0.060469, 0.063906, 0.067344, 0.070781, 0.074219, 0.077656, 0.081094, 0.084531, 0.087969, 0.091406, 0.094844, 0.098281, 0.105469, 0.116406, 0.127344, 0.138281, 0.149219, 0.160156, 0.171094, 0.182031, 0.192969, 0.203906, 0.214844, 0.225781, 0.236719,
 0.247656, 0.258594, 0.269531, 0.280469, 0.291406, 0.302344, 0.313281, 0.324219, 0.335156, 0.346094, 0.357031, 0.367969, 0.378906, 0.389844, 0.400781, 0.411719, 0.422656, 0.433594, 0.444531,
 0.458594, 0.475781, 0.492969, 0.510156, 0.527344, 0.544531, 0.561719, 0.578906, 0.596094, 0.613281, 0.630469, 0.647656, 0.664844, 0.682031, 0.699219, 0.716406, 0.733594, 0.750781, 0.767969,
 0.785156, 0.802344, 0.819531, 0.836719, 0.853906, 0.871094, 0.888281, 0.905469, 0.922656, 0.939844, 0.957031, 0.974219, 0.991406};

  cudaMalloc((void**)&data8bit, 126*sizeof(float));

  cudaMemcpy(data8bit, data, 126*sizeof(float), cudaMemcpyDefault);

  float data_linear[128] = {0.0, 0.007874015748031496, 0.015748031496062992, 0.023622047244094488, 0.031496062992125984,
  			0.03937007874015748, 0.047244094488188976, 0.05511811023622047, 0.06299212598425197, 0.07086614173228346,
  			0.07874015748031496, 0.08661417322834646, 0.09448818897637795, 0.10236220472440945, 0.11023622047244094,
  			0.11811023622047244, 0.12598425196850394, 0.13385826771653542, 0.14173228346456693, 0.14960629921259844,
  			0.15748031496062992, 0.1653543307086614, 0.1732283464566929, 0.18110236220472442, 0.1889763779527559,
  			0.19685039370078738, 0.2047244094488189, 0.2125984251968504, 0.2204724409448819, 0.22834645669291337,
  			0.23622047244094488, 0.2440944881889764, 0.25196850393700787, 0.25984251968503935, 0.26771653543307083,
  			0.2755905511811024, 0.28346456692913385, 0.29133858267716534, 0.2992125984251969, 0.30708661417322836,
  			0.31496062992125984, 0.3228346456692913, 0.3307086614173228, 0.33858267716535434, 0.3464566929133858,
  			0.3543307086614173, 0.36220472440944884, 0.3700787401574803, 0.3779527559055118, 0.3858267716535433,
  			0.39370078740157477, 0.4015748031496063, 0.4094488188976378, 0.41732283464566927, 0.4251968503937008,
  			0.4330708661417323, 0.4409448818897638, 0.44881889763779526, 0.45669291338582674, 0.4645669291338583,
  			0.47244094488188976, 0.48031496062992124, 0.4881889763779528, 0.49606299212598426, 0.5039370078740157,
  			0.5118110236220472, 0.5196850393700787, 0.5275590551181102, 0.5354330708661417, 0.5433070866141733,
  			0.5511811023622047, 0.5590551181102362, 0.5669291338582677, 0.5748031496062992, 0.5826771653543307,
  			0.5905511811023622, 0.5984251968503937, 0.6062992125984252, 0.6141732283464567, 0.6220472440944882,
  			0.6299212598425197, 0.6377952755905512, 0.6456692913385826, 0.6535433070866141, 0.6614173228346456,
  			0.6692913385826772, 0.6771653543307087, 0.6850393700787402, 0.6929133858267716, 0.7007874015748031,
  			0.7086614173228346, 0.7165354330708661, 0.7244094488188977, 0.7322834645669292, 0.7401574803149606,
  			0.7480314960629921, 0.7559055118110236, 0.7637795275590551, 0.7716535433070866, 0.7795275590551181,
  			0.7874015748031495, 0.7952755905511811, 0.8031496062992126, 0.8110236220472441, 0.8188976377952756,
  			0.8267716535433071, 0.8346456692913385, 0.84251968503937, 0.8503937007874016, 0.8582677165354331,
  			0.8661417322834646, 0.8740157480314961, 0.8818897637795275, 0.889763779527559, 0.8976377952755905,
  			0.905511811023622, 0.9133858267716535, 0.9212598425196851, 0.9291338582677166, 0.937007874015748,
  			0.9448818897637795, 0.952755905511811, 0.9606299212598425, 0.968503937007874, 0.9763779527559056,
  			0.984251968503937, 0.9921259842519685, 1.0};

	cudaMalloc((void**)&data8bit_linear, 128*sizeof(float));

	cudaMemcpy(data8bit_linear, data_linear, 128*sizeof(float), cudaMemcpyDefault);

	float data_standard[128] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-06, 2e-06, 3e-06, 4e-06, 5e-06, 6e-06, 7e-06,
					   8e-06, 9e-06, 10e-06, 1e-05, 1.1e-05, 1.2e-05, 1.3e-05, 1.4e-05, 1.5e-05, 2e-05,
					   3e-05, 4e-05, 5e-05, 6e-05, 7e-05, 8e-05, 9e-05, 0.0001, 0.0001,
					   0.00011, 0.00012, 0.00013, 0.00014, 0.00015, 0.0002,
					   0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.001,
					   0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008,
					   0.009, 0.01, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,
					   0.08, 0.09, 0.1, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6,
					   0.7, 0.8, 0.9, 1.0, 1.0, 1.1, 1.20, 1.3, 1.4, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0,
					   7.0, 8.0, 9.0, 10.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0,
					   120.0, 130.0, 140.0, 150.0};

	for(int i =0; i < 128; i++){ data_standard[i] = data_standard[i]*1000.0f; }

	cudaMalloc((void**)&data8bit_standard, 128*sizeof(float));

	cudaMemcpy(data8bit_standard, data_standard, 128*sizeof(float), cudaMemcpyDefault);

	float data_optimal[128] = {0.0, 1.4375e-06, 1.875e-06, 2.3125e-06, 2.75e-06, 3.1875e-06, 3.625e-06, 4.0625e-06, 4.5e-06, 5.175e-06,
			5.85e-06, 6.525e-06, 7.2e-06, 7.875e-06, 8.55e-06, 9.225e-06, 1e-05, 1.4375e-05, 1.875e-05, 2.3125e-05, 2.75e-05, 3.1875e-05,
			3.625e-05, 4.0625e-05, 4.5e-05, 5.175e-05, 5.85e-05, 6.525e-05, 7.2e-05, 7.875e-05, 8.55e-05, 9.225e-05, 0.0001, 0.00014375,
			0.0001875, 0.000231254, 0.000275, 0.00031875, 0.0003625, 0.00040625, 0.00045, 0.0005175, 0.000585, 0.0006525, 0.00072, 0.0007875,
			0.000855, 0.0009225, 0.001, 0.0014375, 0.001875, 0.0023125, 0.00275, 0.0031875, 0.003625, 0.0040625, 0.0045,
			0.005175, 0.00585, 0.006525, 0.0072, 0.007875, 0.00855, 0.009225, 0.01, 0.0143759, 0.01875, 0.023125, 0.0275, 0.031875, 0.03625,
			0.040625, 0.045, 0.05175, 0.0585, 0.06525, 0.072, 0.07875, 0.0855, 0.09225, 0.1, 0.14375, 0.1875, 0.23125, 0.275, 0.31875, 0.3625,
			0.40625, 0.45, 0.5175, 0.585, 0.6525, 0.72, 0.7875, 0.855, 0.9225, 1.0, 1.4375, 1.875, 2.3125, 2.75, 3.1875, 3.625, 4.0625, 4.5, 5.175,
			5.85, 6.525, 7.2, 7.875, 8.55, 9.225, 10.0, 14.375, 18.75, 23.125, 27.5, 31.875, 36.25, 40.625, 45.0, 51.75, 58.5, 65.25, 72.0, 78.75,
			85.5, 92.25};


	for(int i =0; i < 128; i++){ data_optimal[i] = data_optimal[i]*1000.0f; }

	cudaMalloc((void**)&data8bit_optimal, 128*sizeof(float));

	cudaMemcpy(data8bit_optimal, data_optimal, 128*sizeof(float), cudaMemcpyDefault);
}

Weights::~Weights() {
	delete _lrs;
	delete _reducer;
	delete _broadcaster;
    if (_cleanup) {
        delete _hWeights;
        delete _hWeightsInc;
        if (_srcWeights == NULL) {
            delete _weights;
            delete _weightsInc;
            delete _weightsGrad;
        }
    }
}

NVMatrix& Weights::getW() const {
    assert(_onGPU);
    return *_weights;
}

NVMatrix& Weights::getInc() const {
    assert(_onGPU);
    return *_weightsInc;
}

/*
 * TODO: This seems like pretty nasty behavior, I should change this.
 */
NVMatrix& Weights::getGrad() const {
    assert(_onGPU);
    return _useGrad ? *_weightsGrad : *_weightsInc;
}

Matrix& Weights::getCPUW() const {
    return *_hWeights;
}

Matrix& Weights::getCPUWInc() const {
    return *_hWeightsInc;
}

int Weights::getNumRows() const {
    return _hWeights->getNumRows();
}

int Weights::getNumCols() const {
    return _hWeights->getNumCols();
}

map<int,Weights*>& Weights::getReplicas() {
    return _replicas;
}

template<class T> T& Weights::getShard(T& mat, int replicaID) {
    const int n = mat.getNumElements();
    T& line = mat.reshaped(1, n);
    const int shardStart = min(n, replicaID * _shardSize);
    const int shardEnd = min(n, (replicaID + 1) * _shardSize);
    T& slice = line.sliceCols(shardStart, shardEnd);
    assert(slice.isView());
    delete &line;
    return slice;
}

template<class T> T& Weights::getShard(T& mat) {
    return getShard(mat, getReplicaID());
}

ISafeBroadcastNetwork& Weights::getBroadcaster() {
    if (_broadcaster == NULL) {
        set<int> devices;
        for (map<int, Weights*>::const_iterator it = _replicas.begin(); it != _replicas.end(); ++it) {
            devices.insert(it->second->getDeviceID());
        }
        // NOTE: we must use safe broadcaster becasue we want to *add* our value to everyone else
        _broadcaster = &ISafeBroadcastNetwork::make(devices, getDeviceID()); //&(new NaiveBroadcaster(devices, getDeviceID()))->construct();
    }
    return *_broadcaster;
}

IWeightReducer& Weights::getReducer() {
    if (_reducer == NULL) {
        _reducer = &IWeightReducer::make(_replicas, getReplicaID());
    }
    return *_reducer;
}

void Weights::copyToCPU() {
    if (_srcWeights == NULL) {
        assert(_onGPU);
        NVMatrix::syncStream(); // for safety
        if (getReplicaID() == 0) {
            _weights->copyToHost(*_hWeights);

            // Synchronize weights amongst replicas while we're at it.
            map<int,NVMatrix*> weights;
            for (map<int,Weights*>::const_iterator it = _replicas.begin(); it != _replicas.end(); ++it) {
                weights[it->second->getDeviceID()] = &it->second->getW();
            }
            // These things sync before returning.
            getBroadcaster().broadcast(weights, 1, 0);
        }
        if (_useGrad) {
            Matrix& hIncShard = getShard(*_hWeightsInc);
            _weightsInc->copyToHost(hIncShard);
            delete &hIncShard;
        } else { // In this case there's definitely only one replica
            _weightsInc->copyToHost(*_hWeightsInc);
        }
    }
}

// This function is assumed to be called in the order in which the layers
// were defined
void Weights::copyToGPU() {
    assert(!_onGPU);
    // Copies are performed on the default (computation) stream, so that's fine.
    if (_srcWeights == NULL) {
        _weights = _weights == NULL ? new NVMatrix() : _weights;
        _weightsInc = _weightsInc == NULL ? new NVMatrix() : _weightsInc;
        _weights->copyFromHost(*_hWeights, true);

        if (_useGrad) {
            // In this case there is no need to store the entire inc matrix.
            // Just this replica's shard (for synchronization purposes) will do.
            Matrix& hIncShard = getShard(*_hWeightsInc);
            _weightsInc->copyFromHost(hIncShard, true);
            delete &hIncShard;
        } else {
            _weightsInc->copyFromHost(*_hWeightsInc, true);
        }

        _weightsGrad = _useGrad ? (_weightsGrad == NULL ? new NVMatrix(*_weights) : _weightsGrad) : NULL;
    } else {
        _weights = _srcWeights->_weights;
        _weightsInc = _srcWeights->_weightsInc;
        _weightsGrad = _srcWeights->_weightsGrad;
    }
    _onGPU = true;
}

void Weights::aggregateReplicaGradients(float progress) {
    map<int, NVMatrix*> gradShards;
    map<int, NVMatrix*> wShards;
    for (map<int,Weights*>::const_iterator it = _replicas.begin(); it != _replicas.end(); ++it) {
        gradShards[it->first] = &getShard(it->second->getGrad(), getReplicaID());
        wShards[it->first] = &getShard(it->second->getW(), getReplicaID());
        assert(wShards[it->first]->isContiguous() && gradShards[it->first]->isContiguous());


        std::string mat_name = tostr(gradShards[it->first]->getNumRows()) + "x" + tostr(gradShards[it->first]->getNumCols());

        if(!_abs_buffer_map.count(mat_name))
        {
        	cout << "INIT SHARD BUFFER " << mat_name << endl;
			int size = gradShards[it->first]->getNumRows()*gradShards[it->first]->getNumCols();

			cout << size << endl;
			size_t bytes = size*sizeof(unsigned char);
			unsigned char *mat;
			cudaMalloc((void**)&mat, bytes);

			_buffer8bit_map.insert(std::make_pair<std::string, unsigned char*>(mat_name, mat));

			_abs_buffer_map.insert(std::make_pair<std::string, NVMatrix*>(mat_name, new NVMatrix(gradShards[it->first]->getNumRows(), gradShards[it->first]->getNumCols(), false)));

        }

        gradShards[it->first]->abs(*_abs_buffer_map[mat_name]);
		float absMax = (*_abs_buffer_map[mat_name]).max();



		gradShards[it->first]->compress8bit_standard(data8bit_optimal, _buffer8bit_map[mat_name], 0.0000005f*1000.0f, 93.0f*1000.0f);
		gradShards[it->first]->decompress8bit_standard(data8bit_optimal, _buffer8bit_map[mat_name]);
		/*
		gradShards[it->first]->compress8bit_standard(data8bit_standard, _buffer8bit_map[mat_name], 1e-05f*1000.0f, 150.5f*1000.0f);
		gradShards[it->first]->decompress8bit_standard(data8bit_standard, _buffer8bit_map[mat_name]);
*/
		/*
		gradShards[it->first]->compress8bit_linear(data8bit_linear, absMax, _buffer8bit_map[mat_name]);
		gradShards[it->first]->decompress8bit(data8bit_linear, absMax, _buffer8bit_map[mat_name]);


		gradShards[it->first]->compress8bit(data8bit, absMax, _buffer8bit_map[mat_name]);
		gradShards[it->first]->decompress8bit(data8bit, absMax, _buffer8bit_map[mat_name]);
		*/
    }

    float gradScale = _lrs->getValue(progress);
    NVMatrix::setDeviceID(getDeviceID());

    if (_wc > 0) {
        NVMatrixTernaryOps::WeightedAdd wadd = NVMatrixTernaryOps::WeightedAdd(_mom, gradScale, -_wc * _lrs->getValue(progress));
        _weightsInc->applyTernary(wadd, *gradShards[getReplicaID()], *wShards[getReplicaID()], *_weightsInc);
    } else {
        _weightsInc->add(*gradShards[getReplicaID()], _mom, gradScale);
    }

    // Reduce everyone's gradient into my inc shard
    NVMatrix::syncStream(); // Crucial since the reducer does everything in its own streams!!
    getReducer().reduce(gradShards, gradScale, true);

    // Broadcast my inc -> all replicas
    map<int, NVMatrix*> mats; // device id -> grad
    mats[getDeviceID()] = _weightsInc;
    for (map<int, Weights*>::const_iterator it = _replicas.begin(); it != _replicas.end(); ++it) {
        if (it->first != getReplicaID()) {
            mats[it->second->getDeviceID()] = wShards[it->first];
        }
    }
    getBroadcaster().broadcast(mats, 1, 1);

    NVMatrix::setDeviceID(getDeviceID());
    wShards[getReplicaID()]->add(*_weightsInc);

    // Cleanup
    for (map<int,Weights*>::const_iterator it = _replicas.begin(); it != _replicas.end(); ++it) {
        delete gradShards[it->first];
        delete wShards[it->first];
    }
}


// When _useGrad is false, weightsInc is assumed to contain the 
// entire, properly scaled weight increment.
// OTHERWISE, scale your gradient by 1 / numCases only.
// The scaling by epsW will be done in this routine.
void Weights::update(float progress) {
    // Only true owner of weights updates
//    printf("%s update weights\n", _parent->getName().c_str());
    if (_srcWeights == NULL && _lrs->getBaseValue() > 0) {
        assert(_onGPU);
        if (_useGrad) {
            aggregateReplicaGradients(progress);
        } else { // Definitely no replicas in this case
            if (_wc > 0) {
                _weightsInc->add(*_weights, -_wc * _lrs->getValue(progress));
            }

            /*

            if(!_buffer8bit)
            {
            	cout << "INIT BUFFERS" << endl;
				int size = getNumRows()*getNumCols();

				cout << size << endl;
				size_t bytes = size*sizeof(unsigned char);
				cudaMalloc((void**)&_buffer8bit, bytes);

				_abs_buffer = new NVMatrix(getNumRows(), getNumCols(), false);
            }

            _weightsInc->abs(*_abs_buffer);
    		float absMax = (*_abs_buffer).max();



    		cout << "weight " <<  getNumRows() << "x" << getNumCols() << endl;

    		_weightsInc->compress8bit(data8bit, absMax, _buffer8bit);
    		_weightsInc->decompress8bit(data8bit, absMax, _buffer8bit);
    		*/


            _weights->add(*_weightsInc);
        }
        _numUpdates = 0;
    }
}

int Weights::incNumUpdates() {
    if (_srcWeights != NULL) {
        return _srcWeights->incNumUpdates();
    }
    return _numUpdates++;
}

// Returns the number of times a gradient has been computed for this
// weight matrix during the current pass (interval between two calls of update())
// through the net. This number will only be greater than 1 if this weight matrix
// is *shared* by multiple layers in the net.
int Weights::getNumUpdates() const {
    if (_srcWeights != NULL) {
        return _srcWeights->getNumUpdates();
    }
    return _numUpdates;
}

float Weights::getEps(float progress) const {
    return _lrs->getValue(progress);
}

float Weights::getMom() const {
    return _mom;
}

float Weights::getWC() const {
    return _wc;
}

float Weights::getWBall() const {
    return _wball;
}

bool Weights::isUseGrad() const { // is good grammar
    return _useGrad;
}

bool Weights::isOwner() const {
    return _srcWeights == NULL;
}

ParameterSchedule& Weights::getLearningRateSchedule() const {
	return *_lrs;
}

void Weights::addReplica(Weights& replica) {
    _replicas[replica.getReplicaID()] = &replica;

    const int n = _hWeights->getNumElements();
    _shardSize = DIVUP(n, _replicas.size());
}

int Weights::getReplicaID() {
    return _parent->getReplicaID();
}

int Weights::getDeviceID() {
    return _parent->getDeviceID();
}

Layer& Weights::getParent() {
    return *_parent;
}

/* 
 * ===============
 * WeightList
 * ===============
 */
Weights& WeightList::operator[](const int i) const {
    return *_weightList[i];
}

Weights& WeightList::at(const int i) const {
    return *_weightList[i];
}

WeightList::~WeightList() {
    for (int i = 0; i < _weightList.size(); i++) {
        delete _weightList[i];
    }
}

WeightList::WeightList() {
}

void WeightList::addWeights(Weights& w) {
    _weightList.push_back(&w);
}


void WeightList::update(float progress) {
    for (int i = 0; i < getSize(); i++)
    {
        _weightList[i]->update(progress);
    }
}

void WeightList::copyToCPU() {
    for (int i = 0; i < getSize(); i++) {
        _weightList[i]->copyToCPU();
    }
}

void WeightList::copyToGPU() {
    for (int i = 0; i < getSize(); i++) {
        _weightList[i]->copyToGPU();
    }
}

int WeightList::getSize() const {
    return _weightList.size();
}

void WeightList::addReplica(WeightList& replica) {
    for (int i = 0; i < getSize(); i++) {
        _weightList[i]->addReplica(replica[i]);
    }
}
