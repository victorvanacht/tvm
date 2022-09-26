/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/op/contrib/ethosu/convolution.cc
 * \brief Operator definitions for the Arm(R) Ethos(TM)-U NPU convolution ops.
 */


#include "complex.h"

/* still need to check which of the following includes is needed@@@@ */
#include <tvm/ir/error.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/tir/data_layout.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/topi/broadcast.h>
#include <tvm/topi/detail/constant_utils.h>
#include <tvm/topi/elemwise.h>
#include <tvm/topi/nn.h>
#include <tvm/topi/reduction.h>
#include <tvm/topi/transform.h>

#include <vector>

#include "../../../transforms/infer_layout_utils.h"
#include "../../../transforms/pass_utils.h"
#include "../../../transforms/pattern_utils.h"
#include "../../make_op.h"
#include "../../op_common.h"
#include "../../type_relations.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;


namespace tvm {
namespace relay {
namespace op {
namespace contrib {
namespace complex {


// This file is organized according to:
// https://tvm.apache.org/docs/dev/how_to/relay_add_op.html

// Step 1: Defining an Attribute Node


/*! \brief Attributes used in cumsum and cumprod operator */
struct ConjAttrs : public tvm::AttrsNode<ConjAttrs> {
  //Integer axis;
  DataType dtype;
  //Bool exclusive = Bool(false);
  TVM_DECLARE_ATTRS(ConjAttrs, "relay.attrs.ConjAttrs") {
    //TVM_ATTR_FIELD(axis).describe("The axis to operate over").set_default(NullValue<Integer>());
    //TVM_ATTR_FIELD(dtype).describe("Input and output data type").set_default(NullValue<DataType>());
    // Default is 0 which is "false"
    //TVM_ATTR_FIELD(exclusive)
    //    .describe("The first element is not included")
    //    .set_default(Bool(false));
  }
};  // struct ConjAttrs


/*! \brief Attributes used in cumsum and cumprod operator */
struct FFTAttrs : public tvm::AttrsNode<FFTAttrs> {
  //Integer axis;
  //DataType dtype;
  //Bool exclusive = Bool(false);
  TVM_DECLARE_ATTRS(FFTAttrs, "relay.attrs.FFTAttrs") {
    //TVM_ATTR_FIELD(axis).describe("The axis to operate over").set_default(NullValue<Integer>());
    //TVM_ATTR_FIELD(dtype).describe("Input and output data type").set_default(NullValue<DataType>());
    // Default is 0 which is "false"
    //TVM_ATTR_FIELD(exclusive)
    //    .describe("The first element is not included")
    //    .set_default(Bool(false));
  }
};  // struct ConjAttrs

/*! \brief Attributes used in cumsum and cumprod operator */
struct IFFTAttrs : public tvm::AttrsNode<IFFTAttrs> {
  //Integer axis;
  //DataType dtype;
  //Bool exclusive = Bool(false);
  TVM_DECLARE_ATTRS(FFTAttrs, "relay.attrs.IFFTAttrs") {
    //TVM_ATTR_FIELD(axis).describe("The axis to operate over").set_default(NullValue<Integer>());
    //TVM_ATTR_FIELD(dtype).describe("Input and output data type").set_default(NullValue<DataType>());
    // Default is 0 which is "false"
    //TVM_ATTR_FIELD(exclusive)
    //    .describe("The first element is not included")
    //    .set_default(Bool(false));
  }
};  // struct ConjAttrs

/*! \brief Attributes used in cumsum and cumprod operator */
struct FFT2DAttrs : public tvm::AttrsNode<FFT2DAttrs> {
  //Integer axis;
  //DataType dtype;
  //Bool exclusive = Bool(false);
  TVM_DECLARE_ATTRS(FFT2DAttrs, "relay.attrs.FFT2DAttrs") {
    //TVM_ATTR_FIELD(axis).describe("The axis to operate over").set_default(NullValue<Integer>());
    //TVM_ATTR_FIELD(dtype).describe("Input and output data type").set_default(NullValue<DataType>());
    // Default is 0 which is "false"
    //TVM_ATTR_FIELD(exclusive)
    //    .describe("The first element is not included")
    //    .set_default(Bool(false));
  }
};  // struct ConjAttrs

/*! \brief Attributes used in cumsum and cumprod operator */
struct IFFT2DAttrs : public tvm::AttrsNode<IFFT2DAttrs> {
  //Integer axis;
  //DataType dtype;
  //Bool exclusive = Bool(false);
  TVM_DECLARE_ATTRS(FFT2DAttrs, "relay.attrs.IFFT2DAttrs") {
    //TVM_ATTR_FIELD(axis).describe("The axis to operate over").set_default(NullValue<Integer>());
    //TVM_ATTR_FIELD(dtype).describe("Input and output data type").set_default(NullValue<DataType>());
    // Default is 0 which is "false"
    //TVM_ATTR_FIELD(exclusive)
    //    .describe("The first element is not included")
    //    .set_default(Bool(false));
  }
};  // struct ConjAttrs

/*! \brief Attributes used in cumsum and cumprod operator */
struct ConjcvAttrs : public tvm::AttrsNode<ConjcvAttrs> {
  //Integer axis;
  //DataType dtype;
  //Bool exclusive = Bool(false);
  TVM_DECLARE_ATTRS(ConjcvAttrs, "relay.attrs.ConjcvAttrs") {
    //TVM_ATTR_FIELD(axis).describe("The axis to operate over").set_default(NullValue<Integer>());
    //TVM_ATTR_FIELD(dtype).describe("Input and output data type").set_default(NullValue<DataType>());
    // Default is 0 which is "false"
    //TVM_ATTR_FIELD(exclusive)
    //    .describe("The first element is not included")
    //    .set_default(Bool(false));
  }
};  // struct ConjAttrs

// Step 2: Writing a Type Relation

TVM_REGISTER_NODE_TYPE(ConjAttrs); // register the Conjugate operator attributes object
TVM_REGISTER_NODE_TYPE(FFTAttrs); // register the FFT operator attributes object
TVM_REGISTER_NODE_TYPE(IFFTAttrs); // register the IFFT operator attributes object
TVM_REGISTER_NODE_TYPE(FFT2DAttrs); // register the FFT operator attributes object
TVM_REGISTER_NODE_TYPE(IFFT2DAttrs); // register the IFFT operator attributes object
TVM_REGISTER_NODE_TYPE(ConjcvAttrs); // register the Conjcv operator attributes object


bool ConjRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  // types: [data, output]
  ICHECK_EQ(types.size(), 2) << "Expects two types, one for the input and another for the output";
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "Conj: expect input type to be TensorType but get " << types[0];
    return false;
  }


  // Add check here to assert that data type is custom[cmpl]64 @@@@@
  /*
  const auto* param = attrs.as<ConjAttrs>();
  auto dtype = param->dtype;
    if (dtype.is_void()) {
    dtype = data->dtype;
  }
  */

  reporter->Assign(types[1], TensorType(data->shape, data->dtype));
  return true;
}

bool FFTRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  // types: [data, output]
  ICHECK_EQ(types.size(), 2) << "Expects two types, one for the input and another for the output";
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "FFT: expect input type to be TensorType but get " << types[0];
    return false;
  }
  reporter->Assign(types[1], TensorType(data->shape, data->dtype));
  return true;
}

bool IFFTRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  // types: [data, output]
  ICHECK_EQ(types.size(), 2) << "Expects two types, one for the input and another for the output";
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "IFFT: expect input type to be TensorType but get " << types[0];
    return false;
  }
  reporter->Assign(types[1], TensorType(data->shape, data->dtype));
  return true;
}

bool FFT2DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  // types: [data, output]
  ICHECK_EQ(types.size(), 2) << "Expects two types, one for the input and another for the output";
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "FFT2D: expect input type to be TensorType but get " << types[0];
    return false;
  }
  reporter->Assign(types[1], TensorType(data->shape, data->dtype));
  return true;
}

bool IFFT2DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  // types: [data, output]
  ICHECK_EQ(types.size(), 2) << "Expects two types, one for the input and another for the output";
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "IFFT2D: expect input type to be TensorType but get " << types[0];
    return false;
  }
  reporter->Assign(types[1], TensorType(data->shape, data->dtype));
  return true;
}


bool ConjcvRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  // types: [data, output]
  ICHECK_EQ(types.size(), 2) << "Expects two types, one for the input and another for the output";
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    ICHECK(types[0].as<IncompleteTypeNode>())
        << "Conjcv: expect input type to be TensorType but get " << types[0];
    return false;
  }
  reporter->Assign(types[1], TensorType(data->shape, data->dtype));
  return true;
}

// Step 3: Relating the Arity and Attributes to an Operation

RELAY_REGISTER_OP("conj")
    .describe(
        R"doc(Return the complex conjugate of a complex number (custom[cmpl]64))doc" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(3)
    .add_type_rel("Conj", ConjRel)
    //.set_attr<TOpPattern>("TOpPattern", kElemWise);
    .set_attr<TOpPattern>("TOpPattern", kOpaque); //@@@@ should be kElemWise but gives an error during AutoScheduler..... we should check what goes wrong by looking at an example of another elementwise operator such as add, subtract, log, whatever.

RELAY_REGISTER_OP("fft")
    .describe(
        R"doc(Return the complex 1-dimensional FFT of a complex tensor (custom[cmpl]64))doc" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(3)
    .add_type_rel("Fft", FFTRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

RELAY_REGISTER_OP("ifft")
    .describe(
        R"doc(Return the complex 1-dimensional IFFT of a complex tensor (custom[cmpl]64))doc" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(3)
    .add_type_rel("Ifft", IFFTRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);


RELAY_REGISTER_OP("fft2d")
    .describe(
        R"doc(Return the complex 2-dimensional FFT of a complex tensor (custom[cmpl]64))doc" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(3)
    .add_type_rel("Fft2d", FFT2DRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

RELAY_REGISTER_OP("ifft2d")
    .describe(
        R"doc(Return the complex 12dimensional IFFT of a complex tensor (custom[cmpl]64))doc" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(3)
    .add_type_rel("Ifft2d", IFFT2DRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

RELAY_REGISTER_OP("conjcv")
    .describe(
        R"doc(Return the complex conjugate of a complex tensor (custom[cmpl]64))doc" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(3)
    .add_type_rel("Conjcv", ConjcvRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

// Step 4: Implemented in python user code

// Step 5: Implemented in python user code

// Step 6: Creating a Relay Call Node and Exposing a Python Hook

//Expr MakeConj(Expr data, DataType dtype) {
Expr MakeConj(Expr data) {
    auto attrs = make_object<ConjAttrs>();
    //attrs->dtype = dtype;
    static const Op& op = Op::Get("conj");
    return Call(op, {data}, Attrs(attrs), {});
}
TVM_REGISTER_GLOBAL("relay.op._make.conj").set_body_typed(MakeConj);

Expr MakeFFT(Expr data) {
    auto attrs = make_object<FFTAttrs>();
    //attrs->dtype = dtype;
    static const Op& op = Op::Get("fft");
    return Call(op, {data}, Attrs(attrs), {});
}
TVM_REGISTER_GLOBAL("relay.op._make.fft").set_body_typed(MakeFFT);

Expr MakeIFFT(Expr data) {
    auto attrs = make_object<IFFTAttrs>();
    //attrs->dtype = dtype;
    static const Op& op = Op::Get("ifft");
    return Call(op, {data}, Attrs(attrs), {});
}
TVM_REGISTER_GLOBAL("relay.op._make.ifft").set_body_typed(MakeIFFT);

Expr MakeFFT2D(Expr data) {
    auto attrs = make_object<FFT2DAttrs>();
    //attrs->dtype = dtype;
    static const Op& op = Op::Get("fft2d");
    return Call(op, {data}, Attrs(attrs), {});
}
TVM_REGISTER_GLOBAL("relay.op._make.fft2d").set_body_typed(MakeFFT2D);

Expr MakeIFFT2D(Expr data) {
    auto attrs = make_object<IFFT2DAttrs>();
    //attrs->dtype = dtype;
    static const Op& op = Op::Get("ifft2d");
    return Call(op, {data}, Attrs(attrs), {});
}
TVM_REGISTER_GLOBAL("relay.op._make.ifft2d").set_body_typed(MakeIFFT2D);

Expr MakeConjcv(Expr data) {
    auto attrs = make_object<ConjcvAttrs>();
    //attrs->dtype = dtype;
    static const Op& op = Op::Get("conjcv");
    return Call(op, {data}, Attrs(attrs), {});
}
TVM_REGISTER_GLOBAL("relay.op._make.conjcv").set_body_typed(MakeConjcv);


TVM_REGISTER_GLOBAL("tvm.contrib.fft").set_body([](TVMArgs args, TVMRetValue* ret) {
  //std::cout << "Number of arguments:" << args.size() <<std::endl;

  DLTensor* inputTensor  = args[0];
  DLTensor* outputTensor = args[1];

  // calculate number of FFTs (=rows) that we need to compute. Last dimension is the length of the FFT
  const int fftSize=inputTensor->shape[inputTensor->ndim-1];
  int numRows=1;
  for (int i=0; i<(inputTensor->ndim-1); i++) {
    numRows*=inputTensor->shape[i];
  }
  //std::cout << "fftSize:" << fftSize << std::endl;
  //std::cout << "numRows:" << numRows << std::endl;

  /*  
  std::cout << "Argument[0].ndim:" << A->ndim << std::endl;
  std::cout << "Argument[0].shape:" << A->shape[0] << std::endl;

  std::cout << "Argument[1].ndim:" << R->ndim << std::endl;
  std::cout << "Argument[1].shape:" << R->shape[0] << std::endl;

  //std::cout << "Argument[0].strides:" << A->strides[0] << std::endl;
  //std::cout << "Argument[0].byte_offset:" << A->byte_offset << std::endl;

  std::cout << "Argument[0].data[0]" << *((float*)A->data) << std::endl;
  std::cout << "Argument[0].data[1]" << *(((float*)A->data)+1) << std::endl;

  std::cout << "Argument[1].data[0]" << *((float*)R->data) << std::endl;
  std::cout << "Argument[1].data[1]" << *(((float*)R->data)+1) << std::endl;
  */

  Mat inputCV = Mat(numRows,fftSize,CV_32FC2, inputTensor->data);
  Mat outputCV = Mat(numRows,fftSize,CV_32FC2, outputTensor->data);

  //std::cout << inputCV.at<std::complex<float>>(0,0) << "," << inputCV.at<std::complex<float>>(0,1) << "," << inputCV.at<std::complex<float>>(0,2) <<std::endl;

  cv::dft(inputCV, outputCV, DFT_COMPLEX_INPUT + DFT_ROWS , 0); // add DFT_INVERSE flag and DTF_SCALE for iFFT

  //input.at<std::complex<float>>(0,1)=-input.at<std::complex<float>>(0,1);
});

TVM_REGISTER_GLOBAL("tvm.contrib.ifft").set_body([](TVMArgs args, TVMRetValue* ret) {
  
  DLTensor* inputTensor  = args[0];
  DLTensor* outputTensor = args[1];

  // calculate number of IFFTs (=rows) that we need to compute. Last dimension is the length of the IFFT
  const int fftSize=inputTensor->shape[inputTensor->ndim-1];
  int numRows=1;
  for (int i=0; i<(inputTensor->ndim-1); i++) {
    numRows*=inputTensor->shape[i];
  }
  //std::cout << "fftSize:" << fftSize << std::endl;
  //std::cout << "numRows:" << numRows << std::endl;


  Mat inputCV = Mat(numRows,fftSize,CV_32FC2, inputTensor->data);
  Mat outputCV = Mat(numRows,fftSize,CV_32FC2, outputTensor->data);

  cv::dft(inputCV, outputCV, DFT_INVERSE + DFT_SCALE + DFT_COMPLEX_INPUT + DFT_ROWS , 0); // add DFT_INVERSE flag and DTF_SCALE for iFFT
});

TVM_REGISTER_GLOBAL("tvm.contrib.fft2d").set_body([](TVMArgs args, TVMRetValue* ret) {
  //std::cout << "Number of arguments:" << args.size() <<std::endl;

  DLTensor* inputTensor  = args[0];
  DLTensor* outputTensor = args[1];

  // calculate number of 2D FFTs that we need to compute. Last two dimension is the size of the 2D FFT
  const int fftSizeCol=inputTensor->shape[inputTensor->ndim-1];
  const int fftSizeRow=inputTensor->shape[inputTensor->ndim-2];
  const int fftSize = fftSizeCol * fftSizeRow;
  int numFFTs=1;
  for (int i=0; i<(inputTensor->ndim-2); i++) {
    numFFTs*=inputTensor->shape[i];
  }
  //std::cout << "fftSize:" << fftSize << std::endl;
  //std::cout << "numRows:" << numRows << std::endl;

  /*  
  std::cout << "Argument[0].ndim:" << A->ndim << std::endl;
  std::cout << "Argument[0].shape:" << A->shape[0] << std::endl;

  std::cout << "Argument[1].ndim:" << R->ndim << std::endl;
  std::cout << "Argument[1].shape:" << R->shape[0] << std::endl;

  //std::cout << "Argument[0].strides:" << A->strides[0] << std::endl;
  //std::cout << "Argument[0].byte_offset:" << A->byte_offset << std::endl;

  std::cout << "Argument[0].data[0]" << *((float*)A->data) << std::endl;
  std::cout << "Argument[0].data[1]" << *(((float*)A->data)+1) << std::endl;

  std::cout << "Argument[1].data[0]" << *((float*)R->data) << std::endl;
  std::cout << "Argument[1].data[1]" << *(((float*)R->data)+1) << std::endl;
  */
  Mat inputCV;
  Mat outputCV;

  for (int i=0;i<numFFTs; i++) {

    inputCV  = Mat(fftSizeRow, fftSizeCol, CV_32FC2, ((float*)inputTensor ->data) + i * fftSize * 2); // *2 because of 2 floats in 1 complex64
    outputCV = Mat(fftSizeRow, fftSizeCol, CV_32FC2, ((float*)outputTensor->data) + i * fftSize * 2); // *2 because of 2 floats in 1 complex64

    //std::cout << inputCV.at<std::complex<float>>(0,0) << "," << inputCV.at<std::complex<float>>(0,1) << "," << inputCV.at<std::complex<float>>(0,2) <<std::endl;

    cv::dft(inputCV, outputCV, DFT_COMPLEX_INPUT, 0); // add DFT_INVERSE flag and DTF_SCALE for iFFT

  }
  //input.at<std::complex<float>>(0,1)=-input.at<std::complex<float>>(0,1);
});

TVM_REGISTER_GLOBAL("tvm.contrib.ifft2d").set_body([](TVMArgs args, TVMRetValue* ret) {
  
  DLTensor* inputTensor  = args[0];
  DLTensor* outputTensor = args[1];

  // calculate number of 2D FFTs that we need to compute. Last two dimension is the size of the 2D FFT
  const int fftSizeCol=inputTensor->shape[inputTensor->ndim-1];
  const int fftSizeRow=inputTensor->shape[inputTensor->ndim-2];
  const int fftSize = fftSizeCol * fftSizeRow;
  int numFFTs=1;
  for (int i=0; i<(inputTensor->ndim-2); i++) {
    numFFTs*=inputTensor->shape[i];
  }

  Mat inputCV;
  Mat outputCV;

  for (int i=0;i<numFFTs; i++) {

    inputCV  = Mat(fftSizeRow, fftSizeCol, CV_32FC2, ((float*)inputTensor ->data) + i * fftSize * 2); // *2 because of 2 floats in 1 complex64
    outputCV = Mat(fftSizeRow, fftSizeCol, CV_32FC2, ((float*)outputTensor->data) + i * fftSize * 2); // *2 because of 2 floats in 1 complex64

    //std::cout << inputCV.at<std::complex<float>>(0,0) << "," << inputCV.at<std::complex<float>>(0,1) << "," << inputCV.at<std::complex<float>>(0,2) <<std::endl;

    cv::dft(inputCV, outputCV, DFT_INVERSE + DFT_SCALE + DFT_COMPLEX_INPUT, 0); // add DFT_INVERSE flag and DTF_SCALE for iFFT
  }
});


TVM_REGISTER_GLOBAL("tvm.contrib.conjcv").set_body([](TVMArgs args, TVMRetValue* ret) {
  
  DLTensor* inputTensor  = args[0];
  DLTensor* outputTensor = args[1];

  // calculate size of the tensor
  int size=1;
  for (int i=0; i<(inputTensor->ndim); i++) {
    size*=inputTensor->shape[i];
  }
  std::cout << "size:" << size << std::endl;
  

  Mat inputCV = Mat(1,size,CV_32FC2, inputTensor->data);
  Mat outputCV = Mat(1,size,CV_32FC2, outputTensor->data);

  for (int i=0;i<size; i++) {
    outputCV.at<std::complex<float>>(0,i) = std::conj(inputCV.at<std::complex<float>>(0,i));
  }
});






}  // namespace complex
}  // namespace contrib
}  // namespace op
}  // namespace relay
}  // namespace tvm

