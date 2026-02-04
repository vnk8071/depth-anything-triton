#!/usr/bin/env python3
"""
Depth Anything V2 DeepStream Application

Simple script to run depth estimation using NVIDIA DeepStream SDK.
Processes image/video input and saves output as JPEG.

Usage:
    python deepstream_depth.py -i <input> -o <output>
"""

import sys
import os
import argparse
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst


def run_pipeline(input_path, output_path, config_file):
    """Run the DeepStream pipeline"""
    
    Gst.init(None)
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    pipeline_str = (
        f'filesrc location="{input_path}" ! '
        'jpegdec ! '
        'nvvideoconvert ! '
        'video/x-raw(memory:NVMM) ! '
        'mux.sink_0 nvstreammux name=mux batch-size=1 width=1280 height=720 ! '
        f'nvinfer config-file-path="{config_file}" ! '
        'nvvideoconvert ! '
        'video/x-raw ! '
        'videoconvert ! '
        'jpegenc ! '
        f'filesink location="{output_path}"'
    )
    
    print(f"Input:  {input_path}", flush=True)
    print(f"Output: {output_path}", flush=True)
    print(f"Config: {config_file}", flush=True)
    print("Starting pipeline...", flush=True)
    
    try:
        pipeline = Gst.parse_launch(pipeline_str)
    except Exception as e:
        print(f"ERROR: Failed to create pipeline: {e}", flush=True)
        return False
    
    pipeline.set_state(Gst.State.PLAYING)
    
    bus = pipeline.get_bus()
    msg = bus.timed_pop_filtered(
        Gst.CLOCK_TIME_NONE,
        Gst.MessageType.ERROR | Gst.MessageType.EOS
    )
    
    success = True
    if msg:
        if msg.type == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            print(f"ERROR: {err}", flush=True)
            print(f"Debug: {debug}", flush=True)
            success = False
        elif msg.type == Gst.MessageType.EOS:
            print("Pipeline completed successfully", flush=True)
    
    pipeline.set_state(Gst.State.NULL)
    
    if success and os.path.exists(output_path):
        print(f"Output saved to: {output_path}", flush=True)
    
    return success


def main():
    parser = argparse.ArgumentParser(description="Depth Anything V2 with DeepStream")
    parser.add_argument("-i", "--input", required=True, help="Input image file")
    parser.add_argument("-o", "--output", default="output/depth_output.jpg", help="Output file")
    parser.add_argument("-c", "--config", default="config_infer_depth.txt", help="nvinfer config")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}", flush=True)
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}", flush=True)
        sys.exit(1)
    
    success = run_pipeline(args.input, args.output, args.config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
