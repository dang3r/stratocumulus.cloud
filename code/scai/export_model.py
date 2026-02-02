"""
Export trained PyTorch model to ONNX format for web deployment.
"""

import onnx
import torch

from train import get_model


def export_to_onnx(model_path: str, output_path: str):
    """
    Export a trained PyTorch model to ONNX format.

    Args:
        model_path: Path to saved PyTorch model (.pth file)
        output_path: Where to save the ONNX model (.onnx file)

    Steps to implement:
        1. Create model instance using get_model()
        2. Load the trained weights with torch.load() and model.load_state_dict()
        3. Set model to evaluation mode with model.eval()
        4. Create a dummy input tensor (batch_size=1, channels=3, height=224, width=224)
        5. Use torch.onnx.export() with:
           - model
           - dummy input
           - output_path
           - export_params=True
           - opset_version=11 (compatible with onnxruntime-web)
           - input_names=['input']
           - output_names=['output']
           - dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

    Hints:
        - Dummy input shape: torch.randn(1, 3, 224, 224)
        - Wrap export in torch.no_grad() context
        - Print confirmation when export succeeds
    """
    # Load the trained model
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Create dummy input for tracing
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    # Reload and save with embedded data (fixes browser loading issue)
    model_onnx = onnx.load(output_path)
    onnx.save(model_onnx, output_path, save_as_external_data=False)

    print(f"Model exported to {output_path}")

    # Verify the export
    import os

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Model size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    export_to_onnx("best_model.pth", "stratocumulus_model.onnx")
    print("Export complete! Check the file size - should be 2-10MB")
