import torch
import torchvision.models as models
import onnx
import os

onnx_filename = "mobilenetv2.onnx"

# 1. 기존 에러가 발생하는 옛 파일 삭제
if os.path.exists(onnx_filename):
    os.remove(onnx_filename)
    print(f"Removed old {onnx_filename}")

# 2. PyTorch 모델 준비
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

# 3. ONNX Export (Opset 13 사용)
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_filename,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)

# 4. 생성된 ONNX 모델의 IR Version을 강제로 9로 변경 후 재저장
onnx_model = onnx.load(onnx_filename)
print(f"Original IR Version: {onnx_model.ir_version}")

onnx_model.ir_version = 9  # Jetson ONNX Runtime 호환 IR Version (9 이하)

onnx.save(onnx_model, onnx_filename)
print(f"Successfully converted & saved IR Version to {onnx_model.ir_version}")
