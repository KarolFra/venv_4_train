import argparse
import binascii
import sys
from multiprocessing.managers import BaseManager

from video_meas import _capture_worker, _detection_worker


class _VideoManagerClient(BaseManager):
    pass


_VideoManagerClient.register('get_frame_queue')
_VideoManagerClient.register('get_result_queue')
_VideoManagerClient.register('get_stop_event')
_VideoManagerClient.register('get_ai_enabled_event')
_VideoManagerClient.register('get_capture_pause_event')
_VideoManagerClient.register('get_stream_stop_event')


def _parse_args():
    parser = argparse.ArgumentParser(description='Video measurement worker process')
    parser.add_argument('role', choices=['capture', 'detection'])
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--authkey', required=True, help='hex encoded authentication key')
    return parser.parse_args()


def main():
    args = _parse_args()
    try:
        authkey = binascii.unhexlify(args.authkey)
    except (binascii.Error, ValueError) as exc:
        print(f'[video_worker] invalid authkey: {exc}', flush=True)
        return 2
    manager = _VideoManagerClient(address=(args.host, args.port), authkey=authkey)
    try:
        manager.connect()
    except Exception as exc:
        print(f'[video_worker] failed to connect to manager: {exc}', flush=True)
        return 3
    frame_queue = manager.get_frame_queue()
    stop_event = manager.get_stop_event()
    ai_enabled_event = manager.get_ai_enabled_event()
    capture_pause_event = manager.get_capture_pause_event()
    stream_stop_event = manager.get_stream_stop_event()
    if args.role == 'capture':
        _capture_worker(frame_queue, stop_event, capture_pause_event)
    else:
        result_queue = manager.get_result_queue()
        _detection_worker(frame_queue, result_queue, stop_event, ai_enabled_event, capture_pause_event, stream_stop_event)
    return 0


if __name__ == '__main__':
    sys.exit(main())
