#!/usr/bin/env python3
"""
Test script for PointTransformerV2_6C model
This script verifies that the model can be instantiated and run with 6-channel input data.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_pointtransformer_v2_6c():
    """Test PointTransformerV2_6C model instantiation and forward pass"""
    
    print("🧪 Testing PointTransformerV2_6C model...")
    
    try:
        # Import the model
        from mmdet3d.models.backbone.pointtransformer_v2 import PointTransformerV2_6C
        
        print("✅ Successfully imported PointTransformerV2_6C")
        
        # Test model instantiation
        print("🔧 Creating PointTransformerV2_6C model...")
        model = PointTransformerV2_6C(use_precomputed_eigen=True)
        print("✅ Model created successfully")
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"✅ Model moved to {device}")
        
        # Create dummy 6-channel input data
        batch_size = 2
        num_points = 256
        input_data = torch.randn(batch_size, 6, num_points).to(device)  # [B, 6, N]
        
        print(f"📊 Input data shape: {input_data.shape}")
        print(f"📊 Input data device: {input_data.device}")
        
        # Test forward pass
        print("🚀 Running forward pass...")
        model.eval()
        with torch.no_grad():
            xyz, features = model(input_data, num_points)
        
        print(f"✅ Forward pass successful!")
        print(f"📊 Output xyz shape: {xyz.shape}")
        print(f"📊 Output features shape: {features.shape}")
        
        # Verify output shapes
        expected_xyz_shape = (batch_size, 3, num_points)
        expected_features_shape = (batch_size, 64, num_points)  # Assuming 64 output channels
        
        assert xyz.shape == expected_xyz_shape, f"Expected xyz shape {expected_xyz_shape}, got {xyz.shape}"
        assert features.shape == expected_features_shape, f"Expected features shape {expected_features_shape}, got {features.shape}"
        
        print("✅ Output shapes are correct!")
        
        # Test with different batch sizes
        print("🔄 Testing with different batch sizes...")
        for bs in [1, 4, 8]:
            test_data = torch.randn(bs, 6, num_points).to(device)
            with torch.no_grad():
                xyz, features = model(test_data, num_points)
            print(f"   Batch size {bs}: xyz {xyz.shape}, features {features.shape}")
        
        print("✅ All batch size tests passed!")
        
        # Test 3-channel mode (backward compatibility)
        print("🔄 Testing 3-channel mode...")
        model_3c = PointTransformerV2_6C(use_precomputed_eigen=False)
        model_3c = model_3c.to(device)
        
        input_3c = torch.randn(batch_size, 3, num_points).to(device)
        with torch.no_grad():
            xyz_3c, features_3c = model_3c(input_3c, num_points)
        
        print(f"✅ 3-channel mode works: xyz {xyz_3c.shape}, features {features_3c.shape}")
        
        print("\n🎉 All tests passed! PointTransformerV2_6C is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you have the required dependencies installed:")
        print("   - openpoints")
        print("   - mmdet3d")
        return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ed_pointtransformer_v2():
    """Test ED_PointTransformerV2 model instantiation and forward pass"""
    
    print("\n🧪 Testing ED_PointTransformerV2 model...")
    
    try:
        # Import the model
        from mmdet3d.models.backbone.pointtransformer_v2 import ED_PointTransformerV2
        
        print("✅ Successfully imported ED_PointTransformerV2")
        
        # Test model instantiation
        print("🔧 Creating ED_PointTransformerV2 model...")
        model = ED_PointTransformerV2(ED_nsample=10, ED_conv_out=4, use_precomputed_eigen=True)
        print("✅ Model created successfully")
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"✅ Model moved to {device}")
        
        # Create dummy 6-channel input data
        batch_size = 2
        num_points = 256
        input_data = torch.randn(batch_size, 6, num_points).to(device)  # [B, 6, N]
        
        print(f"📊 Input data shape: {input_data.shape}")
        
        # Test forward pass
        print("🚀 Running forward pass...")
        model.eval()
        with torch.no_grad():
            xyz, features = model(input_data, num_points)
        
        print(f"✅ Forward pass successful!")
        print(f"📊 Output xyz shape: {xyz.shape}")
        print(f"📊 Output features shape: {features.shape}")
        
        print("🎉 ED_PointTransformerV2 test passed!")
        return True
        
    except Exception as e:
        print(f"❌ ED_PointTransformerV2 test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Starting PointTransformerV2_6C Model Tests")
    print("=" * 50)
    
    # Test basic PointTransformerV2_6C
    success_1 = test_pointtransformer_v2_6c()
    
    # Test ED_PointTransformerV2
    success_2 = test_ed_pointtransformer_v2()
    
    print("\n" + "=" * 50)
    if success_1 and success_2:
        print("🎉 All tests passed! PointTransformerV2_6C models are ready to use.")
        print("\n📝 Usage examples:")
        print("   Training: ./train_reid.sh 0 pointtransformer_v2_6c reid_nuscenes_pts")
        print("   Testing:  ./test_reid.sh 0 pointtransformer_v2_6c epoch_500 reid_nuscenes_pts")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
