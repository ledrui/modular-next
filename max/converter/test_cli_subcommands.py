#!/usr/bin/env python3
"""
Test script for new CLI subcommand functionality.
"""

import sys
import unittest
from unittest.mock import patch, MagicMock

# Mock the imports that require torch/MAX/HF
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['converter'] = MagicMock()
sys.modules['max.dtype'] = MagicMock()
sys.modules['max.graph'] = MagicMock()
sys.modules['huggingface_hub'] = MagicMock()

# Now import the CLI module
import cli


class TestCLISubcommands(unittest.TestCase):
    
    def test_help_shows_subcommands(self):
        """Test that help shows available subcommands."""
        with patch('sys.argv', ['cli.py', '--help']):
            with self.assertRaises(SystemExit):
                with patch('sys.stdout') as mock_stdout:
                    cli.main()
    
    @patch('cli.cmd_convert')
    def test_convert_subcommand_called(self, mock_convert):
        """Test convert subcommand is called correctly."""
        test_args = [
            'cli.py', 'convert', 'model.pt', 
            '--input-shapes', '1,784', 
            '--output', 'models/'
        ]
        
        with patch('sys.argv', test_args):
            cli.main()
            mock_convert.assert_called_once()
    
    @patch('cli.cmd_download')  
    def test_download_subcommand_called(self, mock_download):
        """Test download subcommand is called correctly."""
        test_args = [
            'cli.py', 'download', 'microsoft/DialoGPT-medium',
            '--output', 'downloads/'
        ]
        
        with patch('sys.argv', test_args):
            cli.main()
            mock_download.assert_called_once()
    
    def test_convert_requires_input_shapes(self):
        """Test that convert command requires input-shapes."""
        test_args = ['cli.py', 'convert', 'model.pt']
        
        with patch('sys.argv', test_args):
            with self.assertRaises(SystemExit):
                cli.main()
    
    def test_convert_help(self):
        """Test convert subcommand help."""
        test_args = ['cli.py', 'convert', '--help']
        
        with patch('sys.argv', test_args):
            with self.assertRaises(SystemExit):
                with patch('sys.stdout') as mock_stdout:
                    cli.main()
    
    def test_download_help(self):
        """Test download subcommand help."""
        test_args = ['cli.py', 'download', '--help']
        
        with patch('sys.argv', test_args):
            with self.assertRaises(SystemExit):
                with patch('sys.stdout') as mock_stdout:
                    cli.main()


class TestInputShapeParsing(unittest.TestCase):
    
    def test_single_shape(self):
        """Test parsing single input shape."""
        result = cli.parse_input_shapes("1,784")
        self.assertEqual(result, [(1, 784)])
    
    def test_multiple_shapes(self):
        """Test parsing multiple input shapes."""
        result = cli.parse_input_shapes("1,784;1,10") 
        self.assertEqual(result, [(1, 784), (1, 10)])
    
    def test_complex_shape(self):
        """Test parsing complex input shape."""
        result = cli.parse_input_shapes("32,3,224,224")
        self.assertEqual(result, [(32, 3, 224, 224)])


if __name__ == "__main__":
    print("Testing CLI subcommand functionality...")
    unittest.main(verbosity=2)