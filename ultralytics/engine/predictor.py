# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run prediction on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ yolo mode=predict model=yolo11n.pt source=0                               # webcam
                                                img.jpg                         # image
                                                vid.mp4                         # video
                                                screen                          # screenshot
                                                path/                           # directory
                                                list.txt                        # list of images
                                                list.streams                    # list of streams
                                                'path/*.jpg'                    # glob
                                                'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP, TCP stream

Usage - formats:
    $ yolo mode=predict model=yolo11n.pt                 # PyTorch
                              yolo11n.torchscript        # TorchScript
                              yolo11n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                              yolo11n_openvino_model     # OpenVINO
                              yolo11n.engine             # TensorRT
                              yolo11n.mlpackage          # CoreML (macOS-only)
                              yolo11n_saved_model        # TensorFlow SavedModel
                              yolo11n.pb                 # TensorFlow GraphDef
                              yolo11n.tflite             # TensorFlow Lite
                              yolo11n_edgetpu.tflite     # TensorFlow Edge TPU
                              yolo11n_paddle_model       # PaddlePaddle
                              yolo11n.mnn                # MNN
                              yolo11n_ncnn_model         # NCNN
                              yolo11n_imx_model          # Sony IMX
                              yolo11n_rknn_model         # Rockchip RKNN
"""

from __future__ import annotations

import platform
import re
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import (
    DEFAULT_CFG,
    LOGGER,
    MACOS,
    WINDOWS,
    callbacks,
    colorstr,
    ops,
)
from ultralytics.utils.checks import check_imgsz, check_imshow
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import (
    attempt_compile,
    select_device,
    smart_inference_mode,
)

STREAM_WARNING = """
inference results will accumulate in RAM unless `stream=True` is passed, causing potential out-of-memory
errors for large sources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict/ for help.

Example:
    results = model(source=..., stream=True)  # generator of Results objects
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs
"""


class BasePredictor:
    """
    A base class for creating predictors.

    This class provides the foundation for prediction functionality, handling model setup, inference,
    and result processing across various input sources.

    Attributes:
        args (SimpleNamespace): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        done_warmup (bool): Whether the predictor has finished setup.
        model (torch.nn.Module): Model used for prediction.
        data (dict): Data configuration.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
        vid_writer (dict[str, cv2.VideoWriter]): Dictionary of {save_path: video_writer} for saving video output.
        plotted_img (np.ndarray): Last plotted image.
        source_type (SimpleNamespace): Type of input source.
        seen (int): Number of images processed.
        windows (list[str]): List of window names for visualization.
        batch (tuple): Current batch data.
        results (list[Any]): Current batch results.
        transforms (callable): Image transforms for classification.
        callbacks (dict[str, list[callable]]): Callback functions for different events.
        txt_path (Path): Path to save text results.
        _lock (threading.Lock): Lock for thread-safe inference.

    Methods:
        preprocess: Prepare input image before inference.
        inference: Run inference on a given image.
        postprocess: Process raw predictions into structured results.
        predict_cli: Run prediction for command line interface.
        setup_source: Set up input source and inference mode.
        stream_inference: Stream inference on input source.
        setup_model: Initialize and configure the model.
        write_results: Write inference results to files.
        save_predicted_images: Save prediction visualizations.
        show: Display results in a window.
        run_callbacks: Execute registered callbacks for an event.
        add_callback: Register a new callback function.
    """

    def __init__(
        self,
        cfg=DEFAULT_CFG,  # 配置文件路径或配置字典，默认使用 YOLOv8 的默认配置
        overrides: dict[str, Any] | None = None,  # 用于覆盖默认配置的参数字典
        _callbacks: (
            dict[str, list[callable]] | None
        ) = None,  # 回调函数字典，用于在推理过程中的关键节点执行自定义逻辑
    ):
        """
        Initialize the BasePredictor class.

        Args:
            cfg (str | dict): Path to a configuration file or a configuration dictionary.
            overrides (dict, optional): Configuration overrides.
            _callbacks (dict, optional): Dictionary of callback functions.
        """
        self.args = get_cfg(
            cfg, overrides
        )  # 将默认配置与用户传入的覆盖参数合并，生成最终的推理配置
        self.save_dir = get_save_dir(self.args)  # 根据配置参数生成结果保存的目录
        if self.args.conf is None:  # 如果用户未指定置信度阈值（conf），则默认设为 0.25
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False  # 标记模型是否完成热身（首次推理前的初始化操作）
        if self.args.show:
            self.args.show = check_imshow(warn=True)  # 检查是否支持图像显示

        # Usable if setup is done
        self.model = None  # 用于存储加载的模型（后续会被赋值为YOLO模型实例）
        self.data = self.args.data  # 数据集配置（如类别信息，从data.yaml中加载）
        self.imgsz = None  # 推理图像尺寸（后续根据输入和模型要求确定）
        self.device = None  # 推理设备（CPU/GPU，后续自动或手动指定）
        self.dataset = None  # 输入数据源（图像/视频/摄像头等，后续解析输入源后赋值）
        self.vid_writer = {}  # 视频写入器字典（key为保存路径，value为视频写入对象）
        self.plotted_img = None  # 存储绘制了检测结果的图像（用于可视化或保存）
        self.source_type = None  # 输入源类型（如图片、视频、摄像头，后续判断）
        self.seen = 0  # 已处理的图像/帧计数（用于统计推理进度）
        self.windows = []  # 存储显示结果的窗口名称（用于多窗口管理）
        self.batch = None  # 存储当前处理的批次数据（图像张量等）
        self.results = None  # 存储推理结果（边界框、类别等结构化数据）
        self.transforms = None  # 图像预处理转换函数（如缩放、归一化）
        self.callbacks = (
            _callbacks or callbacks.get_default_callbacks()
        )  # 初始化回调函数，并添加集成回调。
        self.txt_path = None
        self._lock = threading.Lock()  # for automatic thread-safe inference
        callbacks.add_integration_callbacks(self)

    def preprocess(self, im: torch.Tensor | list[np.ndarray]) -> torch.Tensor:
        """
        Prepare input image before inference.

        Args:
            im (torch.Tensor | list[np.ndarray]): Images of shape (N, 3, H, W) for tensor, [(H, W, 3) x N] for list.

        Returns:
            (torch.Tensor): Preprocessed image tensor of shape (N, 3, H, W).
        """
        # 判断输入是否为张量（非张量则需转换）
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            # 堆叠图像为批量格式（N, H, W, C）
            im = np.stack(self.pre_transform(im))
            # 转换颜色通道（BGR→RGB，因为 OpenCV 读取的是 BGR）
            if im.shape[-1] == 3:
                im = im[..., ::-1]  # BGR to RGB
            # 调整维度顺序（BHWC→BCHW，模型要求的格式）
            im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
            # 确保内存连续（提升计算效率）
            im = np.ascontiguousarray(im)  # contiguous
            # 转换为 PyTorch 张量
            im = torch.from_numpy(im)

        # 将张量迁移到模型运行的设备（CPU/GPU）
        im = im.to(self.device)
        # 转换数据类型（FP16/FP32，与模型保持一致）
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        # 归一化（将 0-255 范围转换为 0.0-1.0）
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    def inference(self, im: torch.Tensor, *args, **kwargs):
        """Run inference on a given image using the specified model and arguments."""
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
            if self.args.visualize and (not self.source_type.tensor)
            else False
        )
        return self.model(
            im,
            augment=self.args.augment,
            visualize=visualize,
            embed=self.args.embed,
            *args,
            **kwargs,
        )

    def pre_transform(self, im: list[np.ndarray]) -> list[np.ndarray]:
        """
        Pre-transform input image before inference.

        Args:
            im (list[np.ndarray]): List of images with shape [(H, W, 3) x N].

        Returns:
            (list[np.ndarray]): List of transformed images.
        """
        same_shapes = len({x.shape for x in im}) == 1  # 检查所有图像是否具有相同尺寸
        # 初始化 LetterBox 工具（用于保持比例缩放图像）
        letterbox = LetterBox(
            self.imgsz,  # 目标尺寸（模型输入尺寸）
            auto=same_shapes
            and self.args.rect
            and (
                self.model.pt
                or (getattr(self.model, "dynamic", False) and not self.model.imx)
            ),
            stride=self.model.stride,  # 模型步长（用于对齐尺寸到步长的倍数）
        )
        return [letterbox(image=x) for x in im]  # 对每个图像应用 LetterBox 变换

    def postprocess(self, preds, img, orig_imgs):
        """Post-process predictions for an image and return them."""
        return preds

    def __call__(self, source=None, model=None, stream: bool = False, *args, **kwargs):
        """
        Perform inference on an image or stream.

        Args:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional):
                Source for inference.
            model (str | Path | torch.nn.Module, optional): Model for inference.
            stream (bool): Whether to stream the inference results. If True, returns a generator.
            *args (Any): Additional arguments for the inference method.
            **kwargs (Any): Additional keyword arguments for the inference method.

        Returns:
            (list[ultralytics.engine.results.Results] | generator): Results objects or generator of Results objects.
        """
        self.stream = stream
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)
        else:
            return list(
                self.stream_inference(source, model, *args, **kwargs)
            )  # merge list of Result into one

    def predict_cli(self, source=None, model=None):
        """
        Method used for Command Line Interface (CLI) prediction.

        This function is designed to run predictions using the CLI. It sets up the source and model, then processes
        the inputs in a streaming manner. This method ensures that no outputs accumulate in memory by consuming the
        generator without storing results.

        Args:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional):
                Source for inference.
            model (str | Path | torch.nn.Module, optional): Model for inference.

        Note:
            Do not modify this function or remove the generator. The generator ensures that no outputs are
            accumulated in memory, which is critical for preventing memory issues during long-running predictions.
        """
        gen = self.stream_inference(source, model)
        for _ in gen:  # sourcery skip: remove-empty-nested-block, noqa
            pass

    def setup_source(self, source):
        """
        Set up source and inference mode.

        Args:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor):
                Source for inference.
        """
        # 1. 验证并调整推理图像尺寸，确保符合模型步长要求
        self.imgsz = check_imgsz(
            self.args.imgsz, stride=self.model.stride, min_dim=2
        )  # check image size
        # 2. 加载输入源，生成数据集对象（统一接口，支持多种输入类型）
        self.dataset = load_inference_source(
            source=source,
            batch=self.args.batch,  # 批次大小
            vid_stride=self.args.vid_stride,  # 视频抽帧步长（间隔多少帧处理一帧）
            buffer=self.args.stream_buffer,  # 是否缓冲流数据
            channels=getattr(self.model, "ch", 3),  # 模型输入通道数（默认3，对应RGB）
        )
        # 3. 记录输入源类型（如stream、video、image等）
        self.source_type = self.dataset.source_type
        # 4. 判断是否为长序列输入（如视频流、大量图像），并给出内存警告
        long_sequence = (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000  # 超过1000张图像视为长序列
            or any(getattr(self.dataset, "video_flag", [False]))  # 包含视频
        )
        if long_sequence:
            import torchvision  # noqa (import here triggers torchvision NMS use in nms.py)

            if not getattr(self, "stream", True):  # 若未启用流式处理，警告内存风险
                LOGGER.warning(STREAM_WARNING)
        # 5. 初始化视频写入器字典（key为保存路径，value为cv2.VideoWriter对象）
        self.vid_writer = {}

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """
        Stream real-time inference on camera feed and save results to file.

        Args:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional):
                Source for inference.
            model (str | Path | torch.nn.Module, optional): Model for inference.
            *args (Any): Additional arguments for the inference method.
            **kwargs (Any): Additional keyword arguments for the inference method.

        Yields:
            (ultralytics.engine.results.Results): Results objects.
        """
        if self.args.verbose:
            LOGGER.info("")

        # 1. 模型初始化
        if not self.model:
            self.setup_model(model)  # 加载模型并配置设备、精度等

        with self._lock:  # 线程安全锁，确保多线程环境下推理稳定
            # 2. 输入源设置
            self.setup_source(
                source if source is not None else self.args.source
            )  # 解析输入源，初始化数据集

            # 3. 结果保存目录准备
            if self.args.save or self.args.save_txt:
                (
                    self.save_dir / "labels" if self.args.save_txt else self.save_dir
                ).mkdir(parents=True, exist_ok=True)

            # 4. 模型预热（首次推理前加速）
            if not self.done_warmup:
                self.model.warmup(
                    imgsz=(
                        1 if self.model.pt or self.model.triton else self.dataset.bs,
                        self.model.ch,
                        *self.imgsz,
                    )
                )
                self.done_warmup = True

            self.seen, self.windows, self.batch = 0, [], None
            # 性能计时器（预处理、推理、后处理三个阶段）
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")  # 触发推理开始回调

            # 5. 流式处理输入数据（核心循环）
            for self.batch in self.dataset:
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch

                # 6. 预处理：图像缩放、格式转换、设备迁移等
                with profilers[0]:
                    im = self.preprocess(im0s)

                # 7. 模型推理
                with profilers[1]:
                    preds = self.inference(
                        im, *args, **kwargs
                    )  # 调用模型得到原始预测结果
                    if self.args.embed:  # 若为特征嵌入模式，直接返回嵌入向量
                        yield from (
                            [preds] if isinstance(preds, torch.Tensor) else preds
                        )  # yield embedding tensors
                        continue

                # 8. 后处理：解析预测结果（如边界框、掩码、类别等）
                with profilers[2]:
                    self.results = self.postprocess(preds, im, im0s)  # 转换为结构化结果
                self.run_callbacks("on_predict_postprocess_end")  # 触发后处理结束回调

                # 9. 结果处理（可视化、保存、日志）
                n = len(im0s)
                try:
                    for i in range(n):
                        self.seen += 1
                        self.results[i].speed = {
                            "preprocess": profilers[0].dt * 1e3 / n,
                            "inference": profilers[1].dt * 1e3 / n,
                            "postprocess": profilers[2].dt * 1e3 / n,
                        }
                        if (
                            self.args.verbose
                            or self.args.save
                            or self.args.save_txt
                            or self.args.show
                        ):
                            s[i] += self.write_results(i, Path(paths[i]), im, s)
                except StopIteration:
                    break

                # 10. 打印批次日志
                if self.args.verbose:
                    LOGGER.info("\n".join(s))

                self.run_callbacks("on_predict_batch_end")  # 触发批次处理结束回调
                yield from self.results

        # 11. 资源释放
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        if self.args.show:
            cv2.destroyAllWindows()  # 关闭所有显示窗口

        # 12. 输出最终性能统计
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)  # speeds per image
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), getattr(self.model, 'ch', 3), *im.shape[2:])}"
                % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = (
                f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}"
                if self.args.save_txt
                else ""
            )
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks("on_predict_end")

    def setup_model(self, model, verbose: bool = True):
        """
        Initialize YOLO model with given parameters and set it to evaluation mode.

        Args:
            model (str | Path | torch.nn.Module, optional): Model to load or use.
            verbose (bool): Whether to print verbose output.
        """
        self.model = AutoBackend(
            model=model or self.args.model,  # 模型路径或已加载的模型实例
            device=select_device(self.args.device, verbose=verbose),  # 选择运行设备
            dnn=self.args.dnn,  # 是否使用 OpenCV DNN 推理（针对 ONNX 等格式）
            data=self.args.data,  # 数据集配置（用于类别信息等）
            fp16=self.args.half,  # 是否启用 FP16 精度推理
            fuse=True,  # 是否融合卷积与 BN 层以加速推理
            verbose=verbose,  # 是否打印模型加载信息
        )
        # 更新设备和精度配置
        self.device = self.model.device  # 同步模型实际使用的设备
        self.args.half = self.model.fp16  # 同步模型实际使用的精度（可能与输入参数不同）
        # 从模型元数据中复用图像尺寸（如果模型是导出的且非动态输入）
        if hasattr(self.model, "imgsz") and not getattr(self.model, "dynamic", False):
            self.args.imgsz = self.model.imgsz  # reuse imgsz from export metadata
        # 设置模型为评估模式
        self.model.eval()
        # 尝试编译模型以优化推理速度（根据设备和配置）
        self.model = attempt_compile(
            self.model, device=self.device, mode=self.args.compile
        )

    def write_results(self, i: int, p: Path, im: torch.Tensor, s: list[str]) -> str:
        """
              将推理结果写入文件或目录。

        参数：
        i (int): 当前图像在批次中的索引。
        p (Path): 当前图像的路径。
        im (torch.Tensor): 预处理后的图像张量。
        s (list[str]): 结果字符串列表。

        返回值：
        (str): 包含结果信息的字符串。
        这个方法负责将检测结果保存到各种格式的文件中，包括：
        文本文件（边界框坐标、类别、置信度）
        图像文件（带标注的可视化结果）
        裁剪的图像区域
        视频文件（对于视频输入）

        同时更新结果字符串列表，用于在控制台显示处理进度和统计信息。
        """
        string = ""  # 用于存储控制台输出的日志字符串
        if len(im.shape) == 3:
            im = im[None]  # 扩展为批量维度（添加 batch 维度，适应批量推理）
        if (
            self.source_type.stream
            or self.source_type.from_img
            or self.source_type.tensor
        ):  # 若输入是流、图像、张量（batch_size >= 1）
            string += f"{i}: "  # 拼接批量索引（如 "0: " 表示第0个样本）
            frame = self.dataset.count  # 从数据集获取当前帧计数（用于视频/流）
        else:
            # 从日志字符串中提取帧索引（针对视频文件，格式如 "frame 5/"）
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None  # 提取帧号（如 5）

        self.txt_path = (
            self.save_dir
            / "labels"
            / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        )
        string += "{:g}x{:g} ".format(*im.shape[2:])  # 拼接图像尺寸（如 "640x480 "）
        result = self.results[i]  # 获取当前帧的推理结果（Results对象）
        result.save_dir = self.save_dir.__str__()  # 为结果对象设置保存目录
        # 拼接检测结果摘要（如 "4 persons, 1 bus"）和推理耗时（如 "79.6ms"）
        string += f"{result.verbose()}{result.speed['inference']:.1f}ms"

        # 若需要保存或显示图像
        if self.args.save or self.args.show:
            self.plotted_img = result.plot(
                line_width=self.args.line_width,  # 检测框线宽
                boxes=self.args.show_boxes,  # 是否显示检测框
                conf=self.args.show_conf,  # 是否显示置信度
                labels=self.args.show_labels,  # 是否显示标签
                im_gpu=None if self.args.retina_masks else im[i],  # 掩码绘制的GPU图像
            )

        # 若启用保存txt
        if self.args.save_txt:
            # 调用Results.save_txt()保存检测结果到txt文件
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        # 若启用保存裁剪图像
        if self.args.save_crop:
            # 保存检测目标的裁剪图像到 save_dir/crops/类别名/ 目录
            result.save_crop(
                save_dir=self.save_dir / "crops", file_name=self.txt_path.stem  # 裁剪图像文件名（与txt一致）
            )
        if self.args.show:
            self.show(str(p))
        if self.args.save:
            # 保存绘制后的图像到 save_dir/ 目录（如 runs/detect/exp1/bus.jpg）
            self.save_predicted_images(self.save_dir / p.name, frame)

        return string

    def save_predicted_images(self, save_path: Path, frame: int = 0):
        """
        Save video predictions as mp4 or images as jpg at specified path.

        Args:
            save_path (Path): Path to save the results.
            frame (int): Frame number for video mode.
        """
        im = self.plotted_img

        # Save videos and streams
        if self.dataset.mode in {"stream", "video"}:
            fps = self.dataset.fps if self.dataset.mode == "video" else 30
            frames_path = (
                self.save_dir / f"{save_path.stem}_frames"
            )  # save frames to a separate directory
            if save_path not in self.vid_writer:  # new video
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                suffix, fourcc = (
                    (".mp4", "avc1")
                    if MACOS
                    else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                )
                self.vid_writer[save_path] = cv2.VideoWriter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc),
                    fps=fps,  # integer required, floats produce error in MP4 codec
                    frameSize=(im.shape[1], im.shape[0]),  # (width, height)
                )

            # Save video
            self.vid_writer[save_path].write(im)
            if self.args.save_frames:
                cv2.imwrite(f"{frames_path}/{save_path.stem}_{frame}.jpg", im)

        # Save images
        else:
            cv2.imwrite(
                str(save_path.with_suffix(".jpg")), im
            )  # save to JPG for best support

    def show(self, p: str = ""):
        """Display an image in a window."""
        im = self.plotted_img  # 获取已绘制推理结果的图像
        if platform.system() == "Linux" and p not in self.windows:
            # 对Linux系统，为新图像创建窗口并设置属性
            self.windows.append(p)
            cv2.namedWindow(
                p, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
            )  # 允许窗口 resize 并保持比例
            cv2.resizeWindow(
                p, im.shape[1], im.shape[0]
            )  # 按图像原始尺寸初始化窗口大小（宽x高）
        # 显示图像
        cv2.imshow(p, im)
        # 处理按键事件：图像模式等待300ms，视频/流模式等待1ms；按q键退出
        if cv2.waitKey(300 if self.dataset.mode == "image" else 1) & 0xFF == ord(
            "q"
        ):  # 300ms if image; else 1ms
            raise StopIteration

    def run_callbacks(self, event: str):
        """Run all registered callbacks for a specific event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func: callable):
        """Add a callback function for a specific event."""
        self.callbacks[event].append(func)
