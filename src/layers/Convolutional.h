#pragma once

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
class ConvolutionalLayer : public Layer {
   public:
    ConvolutionalLayer(const LayerParams inParams, const LayerParams outParams, const LayerParams weightParams, const LayerParams biasParams, 
    const fp32 maxValue, const fp32 minValue, const fp32 scaleValueWeight, const LayerParams sumOfWeights)
        : Layer(inParams, outParams, LayerType::CONVOLUTIONAL),
          weightParam(weightParams),
          weightData(weightParams),
          biasParam(biasParams),
          biasData(biasParams),
          maxValues(maxValue),
          minValues(minValue),
          scaleValueInputs(127.0/(maxValue)),
          scaleValueWeights(127.0/scaleValueWeight),
          zero_points(-127),
          sumOfWeightData(sumOfWeights) {}

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
        sumOfWeightData.loadData();
    }

    // Fre all resources allocated for the layer
    virtual void freeLayer() override {
        Layer::freeLayer();
        weightData.freeData();
        biasData.freeData();
        sumOfWeightData.freeData();
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

    fp32 maxValues;
    fp32 minValues;
    fp32 scaleValueInputs;
    fp32 scaleValueWeights;
    int8_t zero_points;


    LayerData sumOfWeightData;

};

}  // namespace ML