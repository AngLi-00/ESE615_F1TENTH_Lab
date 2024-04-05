import torch
import torch.nn as nn
# import torchvision.models as models
import torch.onnx
import pycuda.driver as cuda
import onnx
import tensorrt as trt
import numpy as np
from PIL import Image

class F110_YOLO(torch.nn.Module):
    def __init__(self):
        super(F110_YOLO, self).__init__()
        # TODO: Change the channel depth of each layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 4, padding = 1, stride = 2)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace = True)

        self.conv2 = nn.Conv2d(16, 32, kernel_size = 4, padding = 1, stride = 2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace = True)

        self.conv3 = nn.Conv2d(32, 64, kernel_size = 4, padding = 1, stride = 2)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace = True)

        self.conv4 = nn.Conv2d(64, 128, kernel_size = 4, padding = 1, stride = 2)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace = True)

        self.conv5 = nn.Conv2d(128, 256, kernel_size = 4, padding = 1, stride = 2)
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace = True)

        self.conv6 = nn.Conv2d(256, 128, kernel_size = 3, padding = 1, stride = 1)
        self.batchnorm6 = nn.BatchNorm2d(128)
        self.relu6 = nn.ReLU(inplace = True)

        self.conv7 = nn.ConvTranspose2d(128, 64, kernel_size = 3, padding = 1, stride = 1)
        self.batchnorm7 = nn.BatchNorm2d(64)
        self.relu7 = nn.ReLU(inplace = True)

        self.conv8 = nn.ConvTranspose2d(64, 32, kernel_size = 3, padding = 1, stride = 1)
        self.batchnorm8 = nn.BatchNorm2d(32)
        self.relu8 = nn.ReLU(inplace = True)

        self.conv9 = nn.Conv2d(32, 5, kernel_size = 1, padding = 0, stride = 1)
        self.relu9 = nn.ReLU()

    def forward(self, x):
        debug = 0 # change this to 1 if you want to check network dimensions
        if debug == 1: print(0, x.shape)
        x = torch.relu(self.batchnorm1(self.conv1(x)))
        if debug == 1: print(1, x.shape)
        x = torch.relu(self.batchnorm2(self.conv2(x)))
        if debug == 1: print(2, x.shape)
        x = torch.relu(self.batchnorm3(self.conv3(x)))
        if debug == 1: print(3, x.shape)
        x = torch.relu(self.batchnorm4(self.conv4(x)))
        if debug == 1: print(4, x.shape)
        x = torch.relu(self.batchnorm5(self.conv5(x)))
        if debug == 1: print(5, x.shape)
        x = torch.relu(self.batchnorm6(self.conv6(x)))
        if debug == 1: print(6, x.shape)
        x = torch.relu(self.batchnorm7(self.conv7(x)))
        if debug == 1: print(7, x.shape)
        x = torch.relu(self.batchnorm8(self.conv8(x)))
        if debug == 1: print(8, x.shape)
        x = self.conv9(x)
        if debug == 1: print(9, x.shape)
        x = torch.cat([x[:, 0:3, :, :], torch.sigmoid(x[:, 3:5, :, :])], dim=1)

        return x

    def get_loss(self, result, truth, lambda_coord = 5, lambda_noobj = 1):
        x_loss = (result[:, 1, :, :] - truth[:, 1, :, :]) ** 2
        y_loss = (result[:, 2, :, :] - truth[:, 2, :, :]) ** 2
        w_loss = (torch.sqrt(result[:, 3, :, :]) - torch.sqrt(truth[:, 3, :, :])) ** 2
        h_loss = (torch.sqrt(result[:, 4, :, :]) - torch.sqrt(truth[:, 4, :, :])) ** 2
        class_loss_obj = truth[:, 0, :, :] * (truth[:, 0, :, :] - result[:, 0, :, :]) ** 2
        class_loss_noobj = (1 - truth[:, 0, :, :]) * lambda_noobj * (truth[:, 0, :, :] - result[:, 0, :, :]) ** 2

        total_loss = torch.sum(lambda_coord * truth[:, 0, :, :] * (x_loss + y_loss + w_loss + h_loss) + class_loss_obj + class_loss_noobj)

        return total_loss


# Step 1: Convert PyTorch model to ONNX
def convert_to_onnx(model_path, output_path):
    # Load the PyTorch model
    model = F110_YOLO().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 180, 320, dtype=torch.float32, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Export the model to ONNX
    torch.onnx.export(model, dummy_input, output_path, input_names=['input'], output_names=['output'])
    print("done converting to onnx")

# # Step 2: Convert ONNX to TensorRT
TRT_LOGGER = trt.Logger()
def build_engine(onnx_file_path, engine_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config = builder.create_builder_config()

    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    config.max_workspace_size = 1 << 30
    # we have only one image in batch
    builder.max_batch_size = 1

    # # use FP16 mode if possible
    # # Comment out this line to enable FP32(default)
    # config.set_flag(trt.BuilderFlag.FP16)

    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')

    # generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    engine = builder.build_engine(network, config)
    print("Completed creating Engine")
    with open(engine_file_path, 'wb') as engine_file:
        engine_file.write(engine.serialize())
    print("TensorRT engine has been converted and saved to", engine_file_path)
    context = engine.create_execution_context()
    return engine, context


# Convert PyTorch model to ONNX
convert_to_onnx('/home/nvidia/lab-7-vision-lab-team-7/model_100.pt', 'model_100.onnx')

# Convert ONNX to TensorRT
build_engine('/home/nvidia/lab-7-vision-lab-team-7/model_100.onnx', 'model_100.trt')