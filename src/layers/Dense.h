#pragma once

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
class DenseLayer : public Layer {
   public:
    DenseLayer(const LayerParams inParams, const LayerParams outParams, const LayerParams weightParams, 
    const LayerParams biasParams, const fp32 max, const fp32 min, const fp32 scaleWeights)
        : Layer(inParams, outParams, LayerType::DENSE),
          weightParam(weightParams),
          weightData(weightParams),
          biasParam(biasParams),
          biasData(biasParams),
          scaleValueInputs(127.0 / max),
          scaleValueWeights(127.0 / scaleWeights),
          maxValue(max),
          minValue(min),
          zero_points(-127) {}

    // Getters
    const LayerParams& getWeightParams() const { return weightParam; }
    const LayerParams& getBiasParams() const { return biasParam; }
    const LayerData& getWeightData() const { return weightData; }
    const LayerData& getBiasData() const { return biasData; }

    // Allocate all resources needed for the layer & Load all of the required data for the layer
    virtual void allocLayer() override {
        Layer::allocLayer();
        weightData.loadData();
        biasData.loadData();
    }

    // Fre all resources allocated for the layer
    virtual void freeLayer() override {
        Layer::freeLayer();
        weightData.freeData();
        biasData.freeData();
    }

    // Virtual functions
    virtual void computeNaive(const LayerData& dataIn) const override;
    virtual void computeThreaded(const LayerData& dataIn) const override;
    virtual void computeTiled(const LayerData& dataIn) const override;
    virtual void computeSIMD(const LayerData& dataIn) const override;
    virtual void computeAccelerated(const LayerData& dataIn) const override;

   private:

    LayerParams weightParam;
    LayerData weightData;

    LayerParams biasParam;
    LayerData biasData;

    fp32 scaleValueInputs;
    fp32 scaleValueWeights;
    fp32 maxValue;
    fp32 minValue;
    int8_t zero_points;
};

}  // namespace ML