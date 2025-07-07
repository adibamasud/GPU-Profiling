# time_measurer_fixed.py - FIXED VERSION
# Fixes applied:
# - Removed duplicate tensor creation (lines 456, 466)
# - Added input shape validation and compatibility testing
# - Added GPU memory cache clearing between measurements
# - Added warnings about model modifications affecting timing
# - Improved error handling and logging
import torch
import torch.nn as nn
import torchvision.models as models
import time
import os
import json
import argparse

# --- Helper Functions (MUST BE IDENTICAL to feature_extractor.py) ---
def get_module_from_target(root_module: nn.Module, target_str: str) -> nn.Module:
    parts = target_str.split('.')
    mod = root_module
    for part in parts:
        mod = getattr(mod, part)
    return mod

def get_model_block_by_name(block_id: str, full_model_instance_dict: dict):
    # --- AlexNet ---
    alexnet = full_model_instance_dict.get('alexnet')
    if alexnet and block_id.startswith("AlexNet_"):
        if block_id == "AlexNet_features_0_Conv": return alexnet.features[0]
        if block_id == "AlexNet_features_1_ReLU": return alexnet.features[1]
        if block_id == "AlexNet_features_2_MaxPool": return alexnet.features[2]
        if block_id == "AlexNet_features_3_Conv": return alexnet.features[3]
        if block_id == "AlexNet_features_4_ReLU": return alexnet.features[4]
        if block_id == "AlexNet_features_5_MaxPool": return alexnet.features[5]
        if block_id == "AlexNet_features_6_Conv": return alexnet.features[6]
        if block_id == "AlexNet_features_7_ReLU": return alexnet.features[7]
        if block_id == "AlexNet_features_8_Conv": return alexnet.features[8]
        if block_id == "AlexNet_features_9_ReLU": return alexnet.features[9]
        if block_id == "AlexNet_features_10_Conv": return alexnet.features[10]
        if block_id == "AlexNet_features_11_ReLU": return alexnet.features[11]
        if block_id == "AlexNet_features_12_MaxPool": return alexnet.features[12]
        if block_id == "AlexNet_AdaptiveAvgPool": return alexnet.avgpool
        if block_id == "AlexNet_classifier_0_Dropout": return alexnet.classifier[0]
        if block_id == "AlexNet_classifier_1_Linear": return alexnet.classifier[1]
        if block_id == "AlexNet_classifier_2_ReLU": return alexnet.classifier[2]
        if block_id == "AlexNet_classifier_3_Dropout": return alexnet.classifier[3]
        if block_id == "AlexNet_classifier_4_Linear": return alexnet.classifier[4]
        if block_id == "AlexNet_classifier_5_ReLU": return alexnet.classifier[5]
        if block_id == "AlexNet_classifier_6_Linear": return alexnet.classifier[6]

    # --- InceptionV3 ---
    inception_v3 = full_model_instance_dict.get('inception_v3')
    if inception_v3 and block_id.startswith("InceptionV3_"):
        if block_id == "InceptionV3_Conv2d_1a_3x3": return inception_v3.Conv2d_1a_3x3
        if block_id == "InceptionV3_Conv2d_2a_3x3": return inception_v3.Conv2d_2a_3x3
        if block_id == "InceptionV3_Conv2d_2b_3x3": return inception_v3.Conv2d_2b_3x3
        if block_id == "InceptionV3_MaxPool_3a_3x3": return inception_v3.maxpool1
        if block_id == "InceptionV3_Conv2d_3b_1x1": return inception_v3.Conv2d_3b_1x1
        if block_id == "InceptionV3_Conv2d_4a_3x3": return inception_v3.Conv2d_4a_3x3
        if block_id == "InceptionV3_MaxPool_5a_3x3": return inception_v3.maxpool2
        if block_id == "InceptionV3_Mixed_5b": return inception_v3.Mixed_5b
        if block_id == "InceptionV3_Mixed_5c": return inception_v3.Mixed_5c
        if block_id == "InceptionV3_Mixed_5d": return inception_v3.Mixed_5d
        if block_id == "InceptionV3_Mixed_6a": return inception_v3.Mixed_6a
        if block_id == "InceptionV3_Mixed_6b": return inception_v3.Mixed_6b
        if block_id == "InceptionV3_Mixed_6c": return inception_v3.Mixed_6c
        if block_id == "InceptionV3_Mixed_6d": return inception_v3.Mixed_6d
        if block_id == "InceptionV3_Mixed_6e": return inception_v3.Mixed_6e
        if block_id == "InceptionV3_AuxLogits": return inception_v3.AuxLogits
        if block_id == "InceptionV3_Mixed_7a": return inception_v3.Mixed_7a
        if block_id == "InceptionV3_Mixed_7b": return inception_v3.Mixed_7b
        if block_id == "InceptionV3_Mixed_7c": return inception_v3.Mixed_7c
        if block_id == "InceptionV3_AdaptiveAvgPool": return inception_v3.avgpool
        if block_id == "InceptionV3_Dropout": return inception_v3.dropout
        if block_id == "InceptionV3_FC_layer": return inception_v3.fc

    # --- ResNet18 ---
    resnet18 = full_model_instance_dict.get('resnet18')
    if resnet18 and block_id.startswith("ResNet18_"):
        if block_id == "ResNet18_conv1": return resnet18.conv1
        if block_id == "ResNet18_bn1": return resnet18.bn1
        if block_id == "ResNet18_relu": return resnet18.relu
        if block_id == "ResNet18_maxpool": return resnet18.maxpool
        if block_id == "ResNet18_layer1_block0": return resnet18.layer1[0]
        if block_id == "ResNet18_layer1_block1": return resnet18.layer1[1]
        if block_id == "ResNet18_layer2_block0": return resnet18.layer2[0]
        if block_id == "ResNet18_layer2_block1": return resnet18.layer2[1]
        if block_id == "ResNet18_layer3_block0": return resnet18.layer3[0]
        if block_id == "ResNet18_layer3_block1": return resnet18.layer3[1]
        if block_id == "ResNet18_layer4_block0": return resnet18.layer4[0]
        if block_id == "ResNet18_layer4_block1": return resnet18.layer4[1]
        if block_id == "ResNet18_avgpool": return resnet18.avgpool
        if block_id == "ResNet18_fc": return resnet18.fc

    # --- ResNet50 ---
    resnet50 = full_model_instance_dict.get('resnet50')
    if resnet50 and block_id.startswith("ResNet50_"):
        if block_id == "ResNet50_conv1": return resnet50.conv1
        if block_id == "ResNet50_bn1": return resnet50.bn1
        if block_id == "ResNet50_relu": return resnet50.relu
        if block_id == "ResNet50_maxpool": return resnet50.maxpool
        # More robust parsing for layer blocks
        if block_id.startswith("ResNet50_layer1_block"): 
            block_num = int(block_id.split('_')[-1].replace('block', ''))
            return resnet50.layer1[block_num]
        if block_id.startswith("ResNet50_layer2_block"): 
            block_num = int(block_id.split('_')[-1].replace('block', ''))
            return resnet50.layer2[block_num]
        if block_id.startswith("ResNet50_layer3_block"): 
            block_num = int(block_id.split('_')[-1].replace('block', ''))
            return resnet50.layer3[block_num]
        if block_id.startswith("ResNet50_layer4_block"): 
            block_num = int(block_id.split('_')[-1].replace('block', ''))
            return resnet50.layer4[block_num]
        if block_id == "ResNet50_avgpool": return resnet50.avgpool
        if block_id == "ResNet50_fc": return resnet50.fc

    # --- MobileNetV2 ---
    mobilenet_v2 = full_model_instance_dict.get('mobilenet_v2')
    if mobilenet_v2 and block_id.startswith("MobileNetV2_"):
        if block_id.startswith("MobileNetV2_features_"):
            block_index = int(block_id.split('_')[2])
            return mobilenet_v2.features[block_index]
        if block_id.startswith("MobileNetV2_Classifier_"):
            if block_id == "MobileNetV2_Classifier_Dropout": return mobilenet_v2.classifier[0]
            if block_id == "MobileNetV2_Classifier_Linear": return mobilenet_v2.classifier[1]

    # --- VGG16 ---
    vgg16 = full_model_instance_dict.get('vgg16')
    if vgg16 and block_id.startswith("VGG16_"):
        if block_id.startswith("VGG16_features_"):
            block_index = int(block_id.split('_')[2])
            return vgg16.features[block_index]
        if block_id.startswith("VGG16_classifier_"):
            block_index = int(block_id.split('_')[2])
            return vgg16.classifier[block_index]
        # VGG16 uses avgpool, not AdaptiveAvgPool by default
        if block_id == "VGG16_avgpool": return vgg16.avgpool

    raise ValueError(f"Unknown block_id: {block_id}. Add it to get_model_block_by_name function or check spelling.")


def measure_execution_time(
        block_identifier: str,
        model: nn.Module,
        example_input: torch.Tensor, 
        target_device: torch.device, 
        warmup_iters=10, 
        actual_iters=25) -> float:
    """A generic timing function that can be used for any module."""
    print(f"Measuring time for block: {block_identifier} on device {target_device}...")

    model_on_device = model.to(target_device)
    example_input_on_device = example_input.to(target_device)
    model_on_device.eval()

    # Warmup
    for _ in range(warmup_iters):
        _ = model_on_device(example_input_on_device)
    if target_device.type == 'cuda': torch.cuda.synchronize()

    # Measurement
    start_times = []
    end_times = []
    for _ in range(actual_iters):
        if target_device.type == 'cuda': torch.cuda.synchronize()
        t_start = time.perf_counter()
        _ = model_on_device(example_input_on_device)
        if target_device.type == 'cuda': torch.cuda.synchronize()
        t_end = time.perf_counter()
        start_times.append(t_start)
        end_times.append(t_end)

    if not start_times: return 0.0

    avg_time_ms = sum([(e - s) * 1000 for s, e in zip(start_times, end_times)]) / len(start_times)
    
    time_data = {
        "block_identifier": block_identifier,
        "target_execution_time_ms": avg_time_ms,
        "target_device": str(target_device),
        "example_input_shape": list(example_input.shape)
    }
    return time_data
    

def estimate_per_call_overhead_ms(target_device: torch.device, iters=100) -> float:
    """
    Estimates the overhead of a single synchronized GPU call by timing a minimal operation.
    This captures the cost of kernel launch + synchronization.
    """
    if target_device.type != 'cuda':
        return 0.0 # Overhead is negligible on CPU

    warmup_iters = 20
    for _ in range(warmup_iters):
        t = torch.empty(1, device=target_device)
        torch.cuda.synchronize()

    start_times = []
    end_times = []
    for _ in range(iters):
        torch.cuda.synchronize() # Ensure GPU is idle before start
        t_start = time.perf_counter()
        torch.empty(1, device=target_device) # The minimal kernel launch
        torch.cuda.synchronize() # Wait for the kernel to complete
        t_end = time.perf_counter()
        start_times.append(t_start)
        end_times.append(t_end)

    avg_time_us = sum([(e - s) * 1_000_000 for s, e in zip(start_times, end_times)]) / len(start_times)
    return avg_time_us / 1000.0

# --- Main Time Measurement Script Logic ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Measure execution time of model blocks on the target device.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save time measurement files.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for time measurement (e.g., 'cpu', 'cuda').")
    parser.add_argument("--model_family", type=str, required=True,
                        choices=["alexnet", "inception_v3", "resnet18", "resnet50", "mobilenet_v2", "vgg16", "generic", "all"],
                        help="Which model family to measure.")
    parser.add_argument("--block_ids", type=str, default=None,
                        help="Comma-separated list of specific block IDs to measure. If None, measures all for the family.")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="If specified, modifies the final layer of loaded models to this many output classes (e.g., 10).")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for the inference timing. Default is 1.")
    parser.add_argument("--compare_sum_vs_full", action="store_true",
                        help="Measure the full model and compare its time against the sum of its individual blocks.")
    parser.add_argument("--adjust_for_overhead", action="store_true",
                        help="When used with --compare_sum_vs_full, estimates and adjusts for per-call overhead.")
    args = parser.parse_args()

    if args.compare_sum_vs_full and (args.model_family == 'all' or args.model_family == 'generic'):
        parser.error("--compare_sum_vs_full can only be used with a specific model family (e.g., resnet50), not 'all' or 'generic'.")
    if args.adjust_for_overhead and not args.compare_sum_vs_full:
        parser.error("--adjust_for_overhead requires --compare_sum_vs_full.")

    TARGET_DEVICE = torch.device(args.device)
    if TARGET_DEVICE.type == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA specified but not available. Switching to CPU.")
        TARGET_DEVICE = torch.device("cpu")

    print(f"Using target measurement device: {TARGET_DEVICE}")
    print(f"Using batch size: {args.batch_size}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading reference model(s) for family: {args.model_family}...")
    full_models_loaded = {}
    if args.model_family == "alexnet" or args.model_family == "all": full_models_loaded['alexnet'] = models.alexnet(weights=None).eval().cpu()
    if args.model_family == "inception_v3" or args.model_family == "all": full_models_loaded['inception_v3'] = models.inception_v3(weights=None).eval().cpu()
    if args.model_family == "resnet18" or args.model_family == "all": full_models_loaded['resnet18'] = models.resnet18(weights=None).eval().cpu()
    if args.model_family == "resnet50" or args.model_family == "all": full_models_loaded['resnet50'] = models.resnet50(weights=None).eval().cpu()
    if args.model_family == "mobilenet_v2" or args.model_family == "all": full_models_loaded['mobilenet_v2'] = models.mobilenet_v2(weights=None).eval().cpu()
    if args.model_family == "vgg16" or args.model_family == "all": full_models_loaded['vgg16'] = models.vgg16(weights=None).eval().cpu()

    # NOTE: Model modifications are applied before timing measurements
    # This may affect performance characteristics, especially for final layers
    if args.num_classes is not None:
        print(f"Modifying final layers to output {args.num_classes} classes...")
        print("WARNING: Model modifications may affect timing measurements!")
        for model_name, model_instance in full_models_loaded.items():
            try:
                if model_name in ['resnet50', 'resnet18', 'inception_v3']:
                    if model_name == 'inception_v3' and hasattr(model_instance, 'AuxLogits') and model_instance.AuxLogits is not None:
                        in_features_aux = model_instance.AuxLogits.fc.in_features
                        model_instance.AuxLogits.fc = nn.Linear(in_features_aux, args.num_classes)
                    in_features = model_instance.fc.in_features
                    model_instance.fc = nn.Linear(in_features, args.num_classes)
                    print(f"  Modified final layer(s) for {model_name}.")
                elif model_name in ['mobilenet_v2', 'vgg16', 'alexnet']:
                    in_features = model_instance.classifier[-1].in_features
                    model_instance.classifier[-1] = nn.Linear(in_features, args.num_classes)
                    print(f"  Modified final layer for {model_name}.")
            except Exception as e:
                print(f"  Error modifying final layer for {model_name}: {e}")
                print(f"  Continuing with original model architecture...")

    # THIS LIST MUST BE IDENTICAL TO THE ONE IN feature_extractor.py
    all_possible_blocks_configs = [
        # --- AlexNet ---
        {"id": "AlexNet_features_0_Conv", "input_shape": (1, 3, 224, 224), "model_source": "alexnet"},
        {"id": "AlexNet_features_1_ReLU", "input_shape": (1, 64, 55, 55), "model_source": "alexnet"},
        {"id": "AlexNet_features_2_MaxPool", "input_shape": (1, 64, 55, 55), "model_source": "alexnet"},
        {"id": "AlexNet_features_3_Conv", "input_shape": (1, 64, 27, 27), "model_source": "alexnet"},
        {"id": "AlexNet_features_4_ReLU", "input_shape": (1, 192, 27, 27), "model_source": "alexnet"},
        {"id": "AlexNet_features_5_MaxPool", "input_shape": (1, 192, 27, 27), "model_source": "alexnet"},
        {"id": "AlexNet_features_6_Conv", "input_shape": (1, 192, 13, 13), "model_source": "alexnet"},
        {"id": "AlexNet_features_7_ReLU", "input_shape": (1, 384, 13, 13), "model_source": "alexnet"},
        {"id": "AlexNet_features_8_Conv", "input_shape": (1, 384, 13, 13), "model_source": "alexnet"},
        {"id": "AlexNet_features_9_ReLU", "input_shape": (1, 256, 13, 13), "model_source": "alexnet"},
        {"id": "AlexNet_features_10_Conv", "input_shape": (1, 256, 13, 13), "model_source": "alexnet"},
        {"id": "AlexNet_features_11_ReLU", "input_shape": (1, 256, 13, 13), "model_source": "alexnet"},
        {"id": "AlexNet_features_12_MaxPool", "input_shape": (1, 256, 13, 13), "model_source": "alexnet"},
        {"id": "AlexNet_AdaptiveAvgPool", "input_shape": (1, 256, 6, 6), "model_source": "alexnet"},
        {"id": "AlexNet_classifier_0_Dropout", "input_shape": (1, 9216), "model_source": "alexnet"},
        {"id": "AlexNet_classifier_1_Linear", "input_shape": (1, 9216), "model_source": "alexnet"},
        {"id": "AlexNet_classifier_2_ReLU", "input_shape": (1, 4096), "model_source": "alexnet"},
        {"id": "AlexNet_classifier_3_Dropout", "input_shape": (1, 4096), "model_source": "alexnet"},
        {"id": "AlexNet_classifier_4_Linear", "input_shape": (1, 4096), "model_source": "alexnet"},
        {"id": "AlexNet_classifier_5_ReLU", "input_shape": (1, 4096), "model_source": "alexnet"},
        {"id": "AlexNet_classifier_6_Linear", "input_shape": (1, 4096), "model_source": "alexnet"},
        # --- InceptionV3 --- (Granularity is by named block, which is standard)
        {"id": "InceptionV3_Conv2d_1a_3x3", "input_shape": (1, 3, 299, 299), "model_source": "inception_v3"},
        {"id": "InceptionV3_Conv2d_2a_3x3", "input_shape": (1, 32, 149, 149), "model_source": "inception_v3"},
        {"id": "InceptionV3_Conv2d_2b_3x3", "input_shape": (1, 32, 147, 147), "model_source": "inception_v3"},
        {"id": "InceptionV3_MaxPool_3a_3x3", "input_shape": (1, 64, 147, 147), "model_source": "inception_v3"},
        {"id": "InceptionV3_Conv2d_3b_1x1", "input_shape": (1, 64, 73, 73), "model_source": "inception_v3"},
        {"id": "InceptionV3_Conv2d_4a_3x3", "input_shape": (1, 80, 73, 73), "model_source": "inception_v3"},
        {"id": "InceptionV3_MaxPool_5a_3x3", "input_shape": (1, 192, 71, 71), "model_source": "inception_v3"},
        {"id": "InceptionV3_Mixed_5b", "input_shape": (1, 192, 35, 35), "model_source": "inception_v3"},
        {"id": "InceptionV3_Mixed_5c", "input_shape": (1, 256, 35, 35), "model_source": "inception_v3"},
        {"id": "InceptionV3_Mixed_5d", "input_shape": (1, 288, 35, 35), "model_source": "inception_v3"},
        {"id": "InceptionV3_Mixed_6a", "input_shape": (1, 288, 35, 35), "model_source": "inception_v3"},
        {"id": "InceptionV3_Mixed_6b", "input_shape": (1, 768, 17, 17), "model_source": "inception_v3"},
        {"id": "InceptionV3_Mixed_6c", "input_shape": (1, 768, 17, 17), "model_source": "inception_v3"},
        {"id": "InceptionV3_Mixed_6d", "input_shape": (1, 768, 17, 17), "model_source": "inception_v3"},
        {"id": "InceptionV3_Mixed_6e", "input_shape": (1, 768, 17, 17), "model_source": "inception_v3"},
        {"id": "InceptionV3_AuxLogits", "input_shape": (1, 768, 17, 17), "model_source": "inception_v3"},
        {"id": "InceptionV3_Mixed_7a", "input_shape": (1, 768, 17, 17), "model_source": "inception_v3"},
        {"id": "InceptionV3_Mixed_7b", "input_shape": (1, 1280, 8, 8), "model_source": "inception_v3"},
        {"id": "InceptionV3_Mixed_7c", "input_shape": (1, 2048, 8, 8), "model_source": "inception_v3"},
        {"id": "InceptionV3_AdaptiveAvgPool", "input_shape": (1, 2048, 8, 8), "model_source": "inception_v3"},
        {"id": "InceptionV3_Dropout", "input_shape": (1, 2048), "model_source": "inception_v3"},
        {"id": "InceptionV3_FC_layer", "input_shape": (1, 2048), "model_source": "inception_v3"},
        # --- ResNet18 ---
        {"id": "ResNet18_conv1", "input_shape": (1, 3, 224, 224), "model_source": "resnet18"},
        {"id": "ResNet18_bn1", "input_shape": (1, 64, 112, 112), "model_source": "resnet18"},
        {"id": "ResNet18_relu", "input_shape": (1, 64, 112, 112), "model_source": "resnet18"},
        {"id": "ResNet18_maxpool", "input_shape": (1, 64, 112, 112), "model_source": "resnet18"},
        {"id": "ResNet18_layer1_block0", "input_shape": (1, 64, 56, 56), "model_source": "resnet18"},
        {"id": "ResNet18_layer1_block1", "input_shape": (1, 64, 56, 56), "model_source": "resnet18"},
        {"id": "ResNet18_layer2_block0", "input_shape": (1, 64, 56, 56), "model_source": "resnet18"},
        {"id": "ResNet18_layer2_block1", "input_shape": (1, 128, 28, 28), "model_source": "resnet18"},
        {"id": "ResNet18_layer3_block0", "input_shape": (1, 128, 28, 28), "model_source": "resnet18"},
        {"id": "ResNet18_layer3_block1", "input_shape": (1, 256, 14, 14), "model_source": "resnet18"},
        {"id": "ResNet18_layer4_block0", "input_shape": (1, 256, 14, 14), "model_source": "resnet18"},
        {"id": "ResNet18_layer4_block1", "input_shape": (1, 512, 7, 7), "model_source": "resnet18"},
        {"id": "ResNet18_avgpool", "input_shape": (1, 512, 7, 7), "model_source": "resnet18"},
        {"id": "ResNet18_fc", "input_shape": (1, 512), "model_source": "resnet18"},
        # --- ResNet50 ---
        {"id": "ResNet50_conv1", "input_shape": (1, 3, 224, 224), "model_source": "resnet50"},
        {"id": "ResNet50_bn1", "input_shape": (1, 64, 112, 112), "model_source": "resnet50"},
        {"id": "ResNet50_relu", "input_shape": (1, 64, 112, 112), "model_source": "resnet50"},
        {"id": "ResNet50_maxpool", "input_shape": (1, 64, 112, 112), "model_source": "resnet50"},
        {"id": "ResNet50_layer1_block0", "input_shape": (1, 64, 56, 56), "model_source": "resnet50"},
        {"id": "ResNet50_layer1_block1", "input_shape": (1, 256, 56, 56), "model_source": "resnet50"},
        {"id": "ResNet50_layer1_block2", "input_shape": (1, 256, 56, 56), "model_source": "resnet50"},
        {"id": "ResNet50_layer2_block0", "input_shape": (1, 256, 56, 56), "model_source": "resnet50"},
        {"id": "ResNet50_layer2_block1", "input_shape": (1, 512, 28, 28), "model_source": "resnet50"},
        {"id": "ResNet50_layer2_block2", "input_shape": (1, 512, 28, 28), "model_source": "resnet50"},
        {"id": "ResNet50_layer2_block3", "input_shape": (1, 512, 28, 28), "model_source": "resnet50"},
        {"id": "ResNet50_layer3_block0", "input_shape": (1, 512, 28, 28), "model_source": "resnet50"},
        {"id": "ResNet50_layer3_block1", "input_shape": (1, 1024, 14, 14), "model_source": "resnet50"},
        {"id": "ResNet50_layer3_block2", "input_shape": (1, 1024, 14, 14), "model_source": "resnet50"},
        {"id": "ResNet50_layer3_block3", "input_shape": (1, 1024, 14, 14), "model_source": "resnet50"},
        {"id": "ResNet50_layer3_block4", "input_shape": (1, 1024, 14, 14), "model_source": "resnet50"},
        {"id": "ResNet50_layer3_block5", "input_shape": (1, 1024, 14, 14), "model_source": "resnet50"},
        {"id": "ResNet50_layer4_block0", "input_shape": (1, 1024, 14, 14), "model_source": "resnet50"},
        {"id": "ResNet50_layer4_block1", "input_shape": (1, 2048, 7, 7), "model_source": "resnet50"},
        {"id": "ResNet50_layer4_block2", "input_shape": (1, 2048, 7, 7), "model_source": "resnet50"},
        {"id": "ResNet50_avgpool", "input_shape": (1, 2048, 7, 7), "model_source": "resnet50"},
        {"id": "ResNet50_fc", "input_shape": (1, 2048), "model_source": "resnet50"},
        # --- MobileNetV2 (Now Complete) ---
        {"id": "MobileNetV2_features_0", "input_shape": (1, 3, 224, 224), "model_source": "mobilenet_v2"},
        {"id": "MobileNetV2_features_1", "input_shape": (1, 32, 112, 112), "model_source": "mobilenet_v2"},
        {"id": "MobileNetV2_features_2", "input_shape": (1, 16, 112, 112), "model_source": "mobilenet_v2"},
        {"id": "MobileNetV2_features_3", "input_shape": (1, 24, 56, 56), "model_source": "mobilenet_v2"},
        {"id": "MobileNetV2_features_4", "input_shape": (1, 24, 56, 56), "model_source": "mobilenet_v2"},
        {"id": "MobileNetV2_features_5", "input_shape": (1, 32, 28, 28), "model_source": "mobilenet_v2"},
        {"id": "MobileNetV2_features_6", "input_shape": (1, 32, 28, 28), "model_source": "mobilenet_v2"},
        {"id": "MobileNetV2_features_7", "input_shape": (1, 32, 28, 28), "model_source": "mobilenet_v2"},
        {"id": "MobileNetV2_features_8", "input_shape": (1, 64, 14, 14), "model_source": "mobilenet_v2"},
        {"id": "MobileNetV2_features_9", "input_shape": (1, 64, 14, 14), "model_source": "mobilenet_v2"},
        {"id": "MobileNetV2_features_10", "input_shape": (1, 64, 14, 14), "model_source": "mobilenet_v2"},
        {"id": "MobileNetV2_features_11", "input_shape": (1, 64, 14, 14), "model_source": "mobilenet_v2"},
        {"id": "MobileNetV2_features_12", "input_shape": (1, 96, 14, 14), "model_source": "mobilenet_v2"},
        {"id": "MobileNetV2_features_13", "input_shape": (1, 96, 14, 14), "model_source": "mobilenet_v2"},
        {"id": "MobileNetV2_features_14", "input_shape": (1, 96, 14, 14), "model_source": "mobilenet_v2"},
        {"id": "MobileNetV2_features_15", "input_shape": (1, 160, 7, 7), "model_source": "mobilenet_v2"},
        {"id": "MobileNetV2_features_16", "input_shape": (1, 160, 7, 7), "model_source": "mobilenet_v2"},
        {"id": "MobileNetV2_features_17", "input_shape": (1, 160, 7, 7), "model_source": "mobilenet_v2"},
        {"id": "MobileNetV2_features_18", "input_shape": (1, 320, 7, 7), "model_source": "mobilenet_v2"},
        # Add the missing final features block (ConvBNActivation)
        {"id": "MobileNetV2_features_19", "input_shape": (1, 1280, 7, 7), "model_source": "mobilenet_v2"},
        {"id": "MobileNetV2_Classifier_Dropout", "input_shape": (1, 1280), "model_source": "mobilenet_v2"},
        {"id": "MobileNetV2_Classifier_Linear", "input_shape": (1, 1280), "model_source": "mobilenet_v2"},
        # --- VGG16 ---
        {"id": "VGG16_features_0", "input_shape": (1, 3, 224, 224), "model_source": "vgg16"},
        {"id": "VGG16_features_1", "input_shape": (1, 64, 224, 224), "model_source": "vgg16"},
        {"id": "VGG16_features_2", "input_shape": (1, 64, 224, 224), "model_source": "vgg16"},
        {"id": "VGG16_features_3", "input_shape": (1, 64, 224, 224), "model_source": "vgg16"},
        {"id": "VGG16_features_4", "input_shape": (1, 64, 224, 224), "model_source": "vgg16"},
        {"id": "VGG16_features_5", "input_shape": (1, 64, 112, 112), "model_source": "vgg16"},
        {"id": "VGG16_features_6", "input_shape": (1, 128, 112, 112), "model_source": "vgg16"},
        {"id": "VGG16_features_7", "input_shape": (1, 128, 112, 112), "model_source": "vgg16"},
        {"id": "VGG16_features_8", "input_shape": (1, 128, 112, 112), "model_source": "vgg16"},
        {"id": "VGG16_features_9", "input_shape": (1, 128, 112, 112), "model_source": "vgg16"},
        {"id": "VGG16_features_10", "input_shape": (1, 128, 56, 56), "model_source": "vgg16"},
        {"id": "VGG16_features_11", "input_shape": (1, 256, 56, 56), "model_source": "vgg16"},
        {"id": "VGG16_features_12", "input_shape": (1, 256, 56, 56), "model_source": "vgg16"},
        {"id": "VGG16_features_13", "input_shape": (1, 256, 56, 56), "model_source": "vgg16"},
        {"id": "VGG16_features_14", "input_shape": (1, 256, 56, 56), "model_source": "vgg16"},
        {"id": "VGG16_features_15", "input_shape": (1, 256, 56, 56), "model_source": "vgg16"},
        {"id": "VGG16_features_16", "input_shape": (1, 256, 56, 56), "model_source": "vgg16"},
        {"id": "VGG16_features_17", "input_shape": (1, 256, 28, 28), "model_source": "vgg16"},
        {"id": "VGG16_features_18", "input_shape": (1, 512, 28, 28), "model_source": "vgg16"},
        {"id": "VGG16_features_19", "input_shape": (1, 512, 28, 28), "model_source": "vgg16"},
        {"id": "VGG16_features_20", "input_shape": (1, 512, 28, 28), "model_source": "vgg16"},
        {"id": "VGG16_features_21", "input_shape": (1, 512, 28, 28), "model_source": "vgg16"},
        {"id": "VGG16_features_22", "input_shape": (1, 512, 28, 28), "model_source": "vgg16"},
        {"id": "VGG16_features_23", "input_shape": (1, 512, 28, 28), "model_source": "vgg16"},
        {"id": "VGG16_features_24", "input_shape": (1, 512, 14, 14), "model_source": "vgg16"},
        {"id": "VGG16_features_25", "input_shape": (1, 512, 14, 14), "model_source": "vgg16"},
        {"id": "VGG16_features_26", "input_shape": (1, 512, 14, 14), "model_source": "vgg16"},
        {"id": "VGG16_features_27", "input_shape": (1, 512, 14, 14), "model_source": "vgg16"},
        {"id": "VGG16_features_28", "input_shape": (1, 512, 14, 14), "model_source": "vgg16"},
        {"id": "VGG16_features_29", "input_shape": (1, 512, 14, 14), "model_source": "vgg16"},
        {"id": "VGG16_features_30", "input_shape": (1, 512, 14, 14), "model_source": "vgg16"},
        # VGG16 uses regular avgpool, not AdaptiveAvgPool
        {"id": "VGG16_avgpool", "input_shape": (1, 512, 7, 7), "model_source": "vgg16"},
        {"id": "VGG16_classifier_0", "input_shape": (1, 25088), "model_source": "vgg16"},
        {"id": "VGG16_classifier_1", "input_shape": (1, 4096), "model_source": "vgg16"},
        {"id": "VGG16_classifier_2", "input_shape": (1, 4096), "model_source": "vgg16"},
        {"id": "VGG16_classifier_3", "input_shape": (1, 4096), "model_source": "vgg16"},
        {"id": "VGG16_classifier_4", "input_shape": (1, 4096), "model_source": "vgg16"},
        {"id": "VGG16_classifier_5", "input_shape": (1, 4096), "model_source": "vgg16"},
        {"id": "VGG16_classifier_6", "input_shape": (1, 4096), "model_source": "vgg16"},
    ]
    per_call_overhead_ms = 0
    if args.device == 'cuda':
        print("\n--- Estimating Per-Call GPU Overhead ---")
        per_call_overhead_ms = estimate_per_call_overhead_ms(TARGET_DEVICE)
        print(f"Estimated per-call launch + sync overhead: {per_call_overhead_ms * 1000:.2f} µs ({per_call_overhead_ms:.4f} ms)")
        print("-" * 40)

    # --- Block Selection ---
    selected_block_configs = []
    if args.block_ids:
        target_block_ids = [s.strip() for s in args.block_ids.split(',')]
        for config in all_possible_blocks_configs:
            if config["id"] in target_block_ids:
                selected_block_configs.append(config)
    else:
        for config in all_possible_blocks_configs:
            if args.model_family == 'all' or config.get("model_source") == args.model_family:
                selected_block_configs.append(config)

    # --- Comparison Logic ---
    full_model_time_ms = 0
    sum_of_blocks_time_ms = 0
    num_blocks_measured = 0

    if args.compare_sum_vs_full:
        print("\n--- Measuring Full Model Execution Time ---")
        model_name = args.model_family
        full_model_instance = full_models_loaded[model_name]
        first_block_config = next(c for c in all_possible_blocks_configs if c['model_source'] == model_name)
        base_input_shape = first_block_config['input_shape']
        actual_input_shape = (args.batch_size, *base_input_shape[1:])
        full_model_input_tensor = torch.randn(actual_input_shape)

        # Clear GPU cache before full model measurement
        if TARGET_DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        full_model_time = measure_execution_time(
            block_identifier='full model',    
            model=full_model_instance,
            example_input=full_model_input_tensor,
            target_device=TARGET_DEVICE
        )
        print(f"Full '{model_name}' model avg time: {full_model_time['target_execution_time_ms']:.4f} ms")
        print(f"Full model input shape: {actual_input_shape}")
        print("-" * 40)

    # --- Main Block-by-Block Measurement Loop ---
    print(f"\n--- Measuring Individual Block Execution Times ---")
    if not selected_block_configs:
        print("Warning: No blocks selected for measurement based on the provided arguments.")
    
    for config in selected_block_configs:
        block_id = config["id"]
        base_shape = config["input_shape"]
        actual_shape = (args.batch_size, *base_shape[1:])

        print(f"Processing block: {block_id}...")
        try:
            model_block_instance = get_model_block_by_name(block_id, full_models_loaded)
        except Exception as e:
            print(f"  Error getting model block: {e}. Skipping.")
            continue
        
        # Validate input shape compatibility
        try:
            example_input_tensor = torch.randn(actual_shape)
            # Test forward pass to ensure shape compatibility
            with torch.no_grad():
                test_output = model_block_instance(example_input_tensor.cpu())
            print(f"  Input shape: {actual_shape}, Output shape: {test_output.shape}")
        except Exception as e:
            print(f"  Error with input shape {actual_shape}: {e}. Skipping.")
            continue

        shape_suffix = "x".join(map(str, base_shape[1:]))
        if not shape_suffix: shape_suffix = str(base_shape[0]) if len(base_shape) == 1 else "scalar"

        block_identifier_for_file = f"{block_id}_bs{args.batch_size}_input{shape_suffix}"

        # Clear GPU cache before measurement for consistent results
        if TARGET_DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        block_time = measure_execution_time(
            block_identifier=block_identifier_for_file,    
            model=model_block_instance,
            example_input=example_input_tensor,
            target_device=TARGET_DEVICE
        )

        if block_time:
            output_filename = os.path.join(args.output_dir, f"{block_identifier_for_file}_time.json")
            with open(output_filename, 'w') as f:
                json.dump(block_time, f, indent=2)
            print(f"  Saved timing to: {output_filename}")

        print(f"  Measured avg time: {block_time['target_execution_time_ms']:.4f} ms")

        if args.compare_sum_vs_full:
            sum_of_blocks_time_ms += block_time['target_execution_time_ms']
            num_blocks_measured += 1
        
        print("-" * 30)

    # --- Final Comparison Report ---
    if args.compare_sum_vs_full:
        adjusted_sum_ms = sum_of_blocks_time_ms - (num_blocks_measured * per_call_overhead_ms)

        print("\n" + "="*60)
        print("      Full Model vs. Sum of Parts Comparison")
        print("="*60)
        print(f"Model Family:            {args.model_family}")
        print(f"Device:                  {TARGET_DEVICE}")
        print(f"Batch Size:              {args.batch_size}\n")
        
        print(f"End-to-End Full Model Time:      {full_model_time['target_execution_time_ms']:10.4f} ms")
        print(f"Sum of Individual Blocks:        {sum_of_blocks_time_ms:10.4f} ms")
        
        if args.adjust_for_overhead:
            print(f"  - Total Estimated Overhead:    ({num_blocks_measured} blocks * {per_call_overhead_ms:.4f} ms/block)")
            print(f"  -                             = {num_blocks_measured * per_call_overhead_ms:7.4f} ms")
            print(f"ADJUSTED Sum of Blocks:          {adjusted_sum_ms:10.4f} ms")
        
        print("-"*60)
        final_sum_to_compare = adjusted_sum_ms if args.adjust_for_overhead else sum_of_blocks_time_ms
        difference = final_sum_to_compare - full_model_time['target_execution_time_ms']
        overhead_percentage = (difference / full_model_time['target_execution_time_ms'] * 100) if full_model_time['target_execution_time_ms'] > 0 else float('inf')
        
        print(f"Difference (Adjusted vs. Full):  {difference:10.4f} ms")
        print(f"Remaining Discrepancy %:         {overhead_percentage:10.2f}%")
        print("="*60)

    print("Time measurement complete.")
