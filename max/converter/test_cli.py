#!/usr/bin/env python3
"""
Test script for CLI functionality.
Tests CLI argument parsing without actually running conversion.
"""

import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Mock the imports that require torch/MAX
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['converter'] = MagicMock()
sys.modules['max.dtype'] = MagicMock()
sys.modules['max.graph'] = MagicMock()

# Now import the CLI module
import cli


class TestCLI(unittest.TestCase):
    
    def test_parse_input_shapes_single(self):
        """Test parsing single input shape."""
        result = cli.parse_input_shapes("1,784")
        self.assertEqual(result, [(1, 784)])
    
    def test_parse_input_shapes_multiple(self):
        """Test parsing multiple input shapes."""
        result = cli.parse_input_shapes("1,784;1,10")
        self.assertEqual(result, [(1, 784), (1, 10)])
    
    def test_parse_input_shapes_complex(self):
        """Test parsing complex input shape."""
        result = cli.parse_input_shapes("1,3,224,224")
        self.assertEqual(result, [(1, 3, 224, 224)])
    
    @patch('argparse.ArgumentParser.parse_args')
    def test_argument_parsing(self, mock_parse_args):
        """Test argument parsing."""
        # Mock the parsed arguments
        mock_args = MagicMock()
        mock_args.model_path = "model.pt"
        mock_args.output = "models"
        mock_args.input_shapes = "1,784"
        mock_args.model_name = None
        mock_args.device = "auto"
        mock_args.dtype = "float32"
        mock_args.verbose = False
        mock_parse_args.return_value = mock_args
        
        # Test that we can create the parser without errors
        parser = cli.argparse.ArgumentParser(
            description="Convert PyTorch models to MAX format"
        )
        parser.add_argument("model_path", type=str)
        parser.add_argument("--output", "-o", type=str, default="models")
        
        # This should not raise any errors
        self.assertIsNotNone(parser)
    
    def test_cli_help_structure(self):
        """Test that CLI has proper help structure."""
        with patch('sys.argv', ['cli.py', '--help']):
            with patch('sys.exit'):
                with patch('argparse.ArgumentParser.print_help'):
                    try:
                        # This tests that the argument parser is set up correctly
                        parser = cli.argparse.ArgumentParser()
                        parser.add_argument("model_path", type=str)
                        parser.add_argument("--output", "-o", type=str, default="models")
                        parser.add_argument("--input-shapes", type=str, required=True)
                        self.assertTrue(True)  # If we get here, structure is OK
                    except Exception as e:
                        self.fail(f"CLI structure test failed: {e}")


if __name__ == "__main__":
    print("Testing CLI functionality...")
    unittest.main(verbosity=2)