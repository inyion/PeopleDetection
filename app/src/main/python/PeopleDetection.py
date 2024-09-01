import abc
import logging
import os
from enum import Enum
from typing import Any, Dict, List, NewType, Tuple, Union

import cv2
import numpy
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow as tf

        tflite = tf.lite
    except ImportError:
        raise ImportError("Failed to load tensorflow")

class EnumInputNodeShapeFormat(Enum):
    NCHW = "nchw"
    NCWH = "ncwh"
    NHWC = "nhwc"
    NWHC = "nwhc"

    CHW = NCHW
    CWH = NCWH
    HWC = NHWC
    WHC = NWHC

    # tensorrt format
    LINEAR = NCHW
    CHW2 = NCHW
    HWC8 = NCHW
    CHW4 = NCHW
    CHW16 = NCHW
    CHW32 = NCHW

    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value):
        value = str(value).upper()
        try:
            return cls[value]
        except KeyError:
            msg = f"{cls.__name__} expected {', '.join(list(cls.__members__.keys()))} but got `{value}`"
            raise KeyError(msg)


class EnumNodeRawDataType(Enum):
    # numpy data type
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "utin32"
    UINT64 = "uint64"
    # BOOL='bool_'

    # openvino data type
    FP16 = FLOAT16
    FP32 = FLOAT32
    FP64 = FLOAT64
    I8 = INT8
    I16 = INT16
    I32 = INT32
    I64 = INT64
    U8 = UINT8
    U16 = UINT16
    U32 = UINT32
    U64 = UINT64

    # tensorrt data type
    FLOAT = FLOAT32
    HALF = FLOAT16

    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value):
        value = str(value).upper()
        try:
            return cls[value]
        except KeyError:
            msg = f"{cls.__name__} expected {', '.join(list(cls.__members__.keys()))} but got `{value}`"
            raise KeyError(msg)


class DataAttribute:
    def __init__(self):
        self.attributes = self.get_props()
        for i in self.attributes:
            setattr(self, f"_{i}", None)

    def __dir__(self):
        return self.attributes

    def __iter__(self):
        for i in dir(self):
            yield i, getattr(self, i)

    def __repr__(self):
        return f"DataAttribute class of '{'location' if self.location is not None else 'name'} {self.key}' layer"

    @property
    def key(self) -> Union[int, str]:
        return self.location if self.location is not None else self.name

    @property
    def shape(self) -> Union[None, Tuple]:
        return self._shape

    @shape.setter
    def shape(self, value: Tuple):
        if not isinstance(value, tuple):
            msg = f"{__class__}.shape should be tuple, but got {type(value)}."
            raise ValueError(msg)
        self._shape = value

    @property
    def dtype(self) -> Union[None, str]:
        return self._dtype

    @dtype.setter
    def dtype(self, value):
        self._dtype = EnumNodeRawDataType(value).value

    @property
    def format(self) -> Union[None, str]:
        return self._format

    @format.setter
    def format(self, value):
        self._format = EnumInputNodeShapeFormat(value).value

    @property
    def location(self) -> Union[None, int]:
        """Union[None, int]: location of the Node"""
        return self._location

    @location.setter
    def location(self, value):
        self._location = value

    @property
    def name(self) -> Union[None, str]:
        """Union[None, str]: name of the Node"""
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def height(self) -> Union[None, int]:
        """Union[None, int]: returns height of the shape."""
        if self._height is None:
            if (self._shape and self._format) is None:
                return None
            if len(self._format) != len(self._shape):
                return None
            self._height = dict(zip(self._format, self._shape)).get("h")
        return self._height

    @property
    def width(self) -> Union[None, int]:
        """Union[None, int]: returns width of the shape."""
        if self._width is None:
            if (self._shape and self._format) is None:
                return None
            if len(self._format) != len(self._shape):
                return None
            self._width = dict(zip(self._format, self._shape)).get("w")
        return self._width

    @classmethod
    def get_props(cls) -> List:
        """returns all class properties as List."""
        return [x for x in dir(cls) if isinstance(getattr(cls, x), property)]


class NotaRuntimeError(Exception):
    """Todo"""

    def __init__(self, msg=""):
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        return f"[+] {self.msg}"


class AlreadyInitializedError(NotaRuntimeError):
    """Todo

    Arguments
    --------
    Class: cls

    """

    def __init__(self, cls=None):
        msg = f"{cls or 'class'} is already initialized. Please call finalize function and retry the initialization."
        super().__init__(msg)


class UninitializedError(NotaRuntimeError):
    """Todo"""

    def __init__(self, cls=None):
        msg = f"{cls or 'class'} is not initialized yet. Please initialize the class."
        super().__init__(msg)


class BasemodelError(NotaRuntimeError):
    """Todo"""


class UnsupportedFunction(NotaRuntimeError):
    """Todo"""


def unsupported_initialize():
    """Todo"""
    raise UnsupportedFunction("'initialize' is not callable from an instance.")


def unsupported_finalize():
    """Todo"""
    raise UnsupportedFunction("'finalize' is not callable from an instance.")


class InvalidPreprocessDataError(NotaRuntimeError):
    """Todo"""


## logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)-8s %(message)s")
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)

## lib.common
_interpreter_dict = {}
INPUT = "input"
OUTPUT = "output"
INTERPRETER = "interpreter"
input_attribute = NewType("input_attribute", Dict[int, DataAttribute])
output_attribute = NewType("output_attribute", Dict[int, DataAttribute])


def is_init(cls) -> bool:
    return True if _interpreter_dict.get(cls, None) is not None else False


def get_input_output_attributes(cls):
    dictionary = _interpreter_dict.get(cls, None)
    if dictionary is None:
        return None, None
    return dictionary.get(INPUT, None), dictionary.get(OUTPUT, None)


def model_finalize(cls: str) -> None:
    if is_init(cls) is False:
        raise UninitializedError
    _interpreter_dict.pop(cls)

class Tflite:
    @staticmethod
    def model_initialize(cls, num_threads: int = 1, **kwargs) -> bool:
        if is_init(cls) is True:
            raise AlreadyInitializedError(cls.__name__)
        file_path = kwargs.get("file")
        with open(os.path.join(os.path.dirname(__file__), file_path), "rb") as f:
            interpreter_obj = tflite.Interpreter(model_content=f.read(), num_threads=num_threads)

        interpreter_obj.allocate_tensors()
        input_attribute, output_attribute = Tflite.model_input_output_attributes(interpreter_obj)

        # global _interpreter_dict
        _interpreter_dict[cls] = {
            INTERPRETER: interpreter_obj,
            INPUT: input_attribute,
            OUTPUT: output_attribute,
        }

    @staticmethod
    def model_input_output_attributes(
            interpreter_object: tflite.Interpreter,
    ) -> Tuple[input_attribute, output_attribute]:
        inputs = {}
        outputs = {}

        for input_detail in interpreter_object.get_input_details():
            input_data_attribute = DataAttribute()
            input_data_attribute.name = input_detail.get("name")
            input_data_attribute.location = input_detail.get("index")
            input_data_attribute.shape = tuple(input_detail.get("shape"))
            input_data_attribute.dtype = input_detail.get("dtype").__name__
            input_data_attribute.format = "nchw" if input_data_attribute.shape[1] == 3 else "nhwc"
            inputs[input_data_attribute.key] = input_data_attribute

        for output_detail in interpreter_object.get_output_details():
            output_data_attribute = DataAttribute()
            output_data_attribute.name = output_detail.get("name")
            output_data_attribute.location = output_detail.get("index")
            output_data_attribute.shape = tuple(output_detail.get("shape"))
            output_data_attribute.dtype = output_detail.get("dtype").__name__
            outputs[output_data_attribute.key] = output_data_attribute

        return inputs, outputs

    @staticmethod
    def model_inference(cls: str, preprocess_result: Dict[int, numpy.ndarray], **kwargs) -> Dict[int, numpy.ndarray]:
        interpreter_dict = _interpreter_dict.get(cls, None)
        if interpreter_dict is None:
            raise UninitializedError
        interpreter_obj = interpreter_dict.get(INTERPRETER)
        output: Dict[int, DataAttribute] = interpreter_dict.get(OUTPUT)

        for location, value in iter(preprocess_result.items()):
            interpreter_obj.set_tensor(location, value)
        interpreter_obj.invoke()

        output_dict = {}
        for output_location in iter(output):
            output_dict[output_location] = interpreter_obj.get_tensor(output_location)

        return output_dict


class Basemodel(metaclass=abc.ABCMeta):
    @classmethod
    def initialize(cls, **kwargs) -> None:
        Tflite.model_initialize(cls=cls, **kwargs)

    def __new__(cls):
        if is_init(cls) is False:
            raise UninitializedError
        obj = super().__new__(cls)
        setattr(obj, "initialize", unsupported_initialize)
        setattr(obj, "finalize", unsupported_finalize)
        return obj

    @property
    def inputs(cls) -> Union[None, Dict[int, DataAttribute]]:
        input, _ = get_input_output_attributes(cls.__class__)
        return input

    @property
    def outputs(cls) -> Union[None, Dict[int, DataAttribute]]:
        _, output = get_input_output_attributes(cls.__class__)
        return output

    @classmethod
    def initialize(cls, **kwargs) -> None:
        """Todo"""
        Tflite.model_initialize(cls=cls, **kwargs)

    @classmethod
    def finalize(cls, **kwargs) -> None:
        """Todo"""
        model_finalize(cls=cls, **kwargs)

    def preprocess(self, input_data: Union[str, Any]) -> Dict[Union[str, int], numpy.ndarray]:
        if isinstance(input_data, str) is True:
            return {next(iter(self.inputs)): cv2.imread(input_data)}
        else:
            return {next(iter(self.inputs)): input_data}

    def postprocess(self, inference_result: numpy.ndarray) -> Any:
        return inference_result

    def run(self, input_data: Any, **kwargs):
        pre_result = self.preprocess(input_data)
        inference_result = Tflite.model_inference(cls=self.__class__, preprocess_result=pre_result)
        post_result = self.postprocess(inference_result)
        return post_result


class PeopleDetectionModel(Basemodel):

    def __init__(self):
        self.classes = ['Person']
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.anchor = [[[[[[12.0, 18.0]]],[[[37.0, 49.0]]],[[[52.0, 132.0]]]]],[[[[[115.0, 73.0]]],[[[119.0, 199.0]]],[[[242.0, 238.0]]]]]]
        self.stride = [16.0, 32.0]

        self.input_layer = list(self.inputs.keys())
        if len(self.input_layer) > 1:
            print(f'This model gets {len(self.input_layer)} nodes')

        self.output_layer = list(self.outputs.keys())
        if len(self.output_layer) > 1:
            print(f'This model outputs {len(self.output_layer)} nodes')

    def preprocess(self, image) -> Dict[int, np.ndarray]:
        preprocessed_data = {}

        # **수정된 부분: 이미지가 None인지 확인**
        if image is None:
            raise ValueError("Failed to load image. Please check the file path or the file itself.")

        for key in self.input_layer:
            input_attribute = self.inputs.get(key)
            self.input_size = [input_attribute.height, input_attribute.width]

            origin_h, origin_w, origin_c = image.shape  # 여기서 오류가 발생할 수 있음
            self.origin_h, self.origin_w = origin_h, origin_w
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 기존 이미지 전처리 로직들...

            if input_attribute.format == 'nchw':
                data = data.transpose(0,3,1,2)
            preprocessed_data[key] = data
        return preprocessed_data

    def make_grid(self, nx=20, ny=20):
        yv, xv = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        grid = np.stack((xv, yv), axis=-1).reshape((1, 1, ny, nx, 2)).astype(np.float32)
        return grid

    def postprocess(self, inference_results):
        grid = [np.zeros((1, 1), dtype=np.float32) for _ in range(2)]
        predections = None
        for index, key in enumerate(self.output_layer):
            inference_data = inference_results.get(key)
            bs, ch, ny, nx, _ = inference_data.shape

            if grid[index].shape[2:4] != inference_data.shape[2:4] or False:
                grid[index] = self.make_grid(nx, ny)

            y = self.sigmoid_5d(np.array([inference_data]))
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[index]) * self.stride[index]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) * (y[..., 2:4] * 2) * self.anchor[index]  # wh
            candidates = y.reshape((-1, len(self.classes)+5))
            if predections is None:
                predections = candidates
            else:
                predections = np.concatenate((predections, candidates), axis=0)

        # result = result.reshape((-1, len(self.classes)+5))
        result = self.nms(predections, self.conf_thres, self.iou_thres)
        result = self.normalize(result)
        result = self.scale_coords(result)
        self.print_result(result)
        return result

    def sigmoid_5d(self, x):
        result = np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    for l in range(x.shape[3]):
                        for m in range(x.shape[4]):
                            result[i,j,k,l,m] = 1 / (1 + np.exp(-(x[i,j,k,l,m])))
        return result

    def nms(self, prediction, conf_thres, iou_thres):
        prediction = prediction[prediction[..., 4] > conf_thres]
        boxes = self.xywh2xyxy(prediction[:, :4])
        res = self.non_max_suppression(boxes, prediction[:, 4], iou_thres)
        result_boxes = []
        for r in res:
            tmp = np.zeros(6)
            j = prediction[r, 5:].argmax()
            tmp[0] = boxes[r][0].item()
            tmp[1] = boxes[r][1].item()
            tmp[2] = boxes[r][2].item()
            tmp[3] = boxes[r][3].item()
            tmp[4] = prediction[r][4].item()
            tmp[5] = j
            result_boxes.append(tmp)
        return result_boxes

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

        return y

    def non_max_suppression(self, boxes, scores, iou_thres):
        assert boxes.shape[0] == scores.shape[0]
        # bottom-left origin
        ys1 = boxes[:, 0]
        xs1 = boxes[:, 1]
        # top-right target
        ys2 = boxes[:, 2]
        xs2 = boxes[:, 3]
        # box coordinate ranges are inclusive-inclusive
        areas = (ys2 - ys1) * (xs2 - xs1)
        scores_indexes = scores.argsort().tolist()
        boxes_keep_index = []
        while len(scores_indexes):
            index = scores_indexes.pop()
            boxes_keep_index.append(index)
            if not len(scores_indexes):
                break
            ious = self.compute_iou(boxes[index], boxes[scores_indexes], areas[index], areas[scores_indexes])
            filtered_indexes = np.where(ious > iou_thres)[0]
            # if there are no more scores_index
            # then we should pop it
            scores_indexes = [
                v for (i, v) in enumerate(scores_indexes)
                if i not in filtered_indexes
            ]
        return np.array(boxes_keep_index)

    def compute_iou(self, box, boxes, box_area, boxes_area):
        assert boxes.shape[0] == boxes_area.shape[0]
        ys1 = np.maximum(box[0], boxes[:, 0])
        xs1 = np.maximum(box[1], boxes[:, 1])
        ys2 = np.minimum(box[2], boxes[:, 2])
        xs2 = np.minimum(box[3], boxes[:, 3])
        intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
        unions = box_area + boxes_area - intersections
        ious = intersections / unions
        return ious

    def scale_coords(self, coords):
        if len(coords)==0:
            return coords
        gain = min(self.input_size[0] / self.origin_h, self.input_size[1] / self.origin_w)  # gain  = old / new
        pad = (self.input_size[0] - self.origin_w * gain) / 2, (self.input_size[1] - self.origin_h * gain) / 2  # wh padding
        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords)
        return coords

    def clip_coords(self, coords):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, self.origin_w)  # x1, x2
        coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, self.origin_h)  # y1, y2

    def normalize(self, boxes):
        if not boxes:
            return boxes
        np_boxes = np.array(boxes)

        if np.all(np_boxes[:,:4] <= 1.0):
            # restore result
            for box in boxes:
                box[0] *= self.input_size[1]
                box[1] *= self.input_size[0]
                box[2] *= self.input_size[1]
                box[3] *= self.input_size[0]
            return np.array(boxes)

        return np.array(boxes)

    def print_result(self, result_label):
        print("--------------------------------------------------------------")
        if result_label is None or len(result_label) == 0:
            print(' - Nothing Detected!')
        else:
            for i, label in enumerate(result_label):
                detected = str(self.classes[int(label[5])])
                conf_score = label[4]
                x1, y1, x2, y2 = label[0], label[1], label[2], label[3]
                print(' - Object {}'.format(i+1))
                print('     - CLASS : {}'.format(detected))
                print('     - SCORE : {:5.4f}'.format(conf_score))
                print('     - BOXES : {:6.2f} {:6.2f} {:6.2f} {:6.2f}'.format(x1,y1,x2,y2))
        print("--------------------------------------------------------------\n")


PeopleDetectionModel.initialize(framework="tflite", file="model.tflite")
k = PeopleDetectionModel()
image = cv2.imread("example.jpg")
# **이미지 로드 성공 여부 확인 추가**
if image is None:
    print("Error: Image could not be loaded. Check if the file path is correct.")
else:
    k.run(image)
PeopleDetectionModel.finalize()
