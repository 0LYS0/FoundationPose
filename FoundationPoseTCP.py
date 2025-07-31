# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse

import threading
from threading import Thread, Lock
import json
import os

HEADER = b"<BEGIN>"
FOOTER = b"<END>"
SIZE_FORMAT = "Q"  # 8-byte unsigned long long (for length)
DIR_PATH = os.path.dirname(os.path.abspath(__file__))

import socket
import struct
import pickle
import threading


class FoundationPoseCore:
    def __init__(self, color, depth, mask, K, mesh_file):
        code_dir = DIR_PATH
        # mesh_file = f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj'
        # mesh_file = '/home/rnbmarch/Desktop/dishware/ver_B/13/13.obj'

        self._est_refine_iter = 5
        self._track_refine_iter = 2
        self._debug = 1
        self._debug_dir = f'{code_dir}/debug'

        os.system(f'rm -rf {self._debug_dir}/* && mkdir -p {self._debug_dir}/track_vis {self._debug_dir}/ob_in_cam')

        set_logging_format()
        set_seed(0)

        self._mesh = trimesh.load(mesh_file)
        self._to_origin, extents = trimesh.bounds.oriented_bounds(self._mesh)
        self._bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        self._scorer = ScorePredictor()
        self._refiner = PoseRefinePredictor()
        self._glctx = dr.RasterizeCudaContext()
        self._est = FoundationPose(model_pts=self._mesh.vertices, model_normals=self._mesh.vertex_normals, mesh=self._mesh, scorer=self._scorer,
                             refiner=self._refiner, debug_dir=self._debug_dir, debug=self._debug, glctx=self._glctx)
        self._init_estimator(color, depth, mask, K)

        logging.info("estimator initialization done")

    def _init_estimator(self, color:np.ndarray, depth:np.ndarray, mask:np.ndarray, K:np.ndarray):
        '''
        color: (H, W, 3)
        depth: (H, W)
        mask: (H, W)
        K: (3, 3)
        '''
        mask = mask.copy().astype(bool)
        pose = self._est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=self._est_refine_iter)

        if self._debug >= 3:
            m = mesh.copy()
            m.apply_transform(pose)
            m.export(f'{self._debug_dir}/model_tf.obj')
            xyz_map = depth2xyzmap(depth, reader.K)
            valid = depth >= 0.001
            pcd = toOpen3dCloud(xyz_map[valid], color[valid])
            o3d.io.write_point_cloud(f'{self._debug_dir}/scene_complete.ply', pcd)


    def inference(self, color:np.ndarray, depth:np.ndarray, K:np.ndarray):

        pose = self._est.track_one(rgb=color, depth=depth, K=K, iteration=self._track_refine_iter)

        # os.makedirs(f'{self._debug_dir}/ob_in_cam', exist_ok=True)
        # np.savetxt(f'{self._debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4, 4))

        if self._debug >= 1:
            center_pose = pose @ np.linalg.inv(self._to_origin)
            vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=self._bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0,
                                is_input_rgb=True)

            if self._debug >= 2:
                os.makedirs(f'{self._debug_dir}/track_vis', exist_ok=True)
                imageio.imwrite(f'{self._debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

        else:
            vis = color.copy()

        return pose.reshape(4, 4), vis

class FoundationPoseTCPServer:
    def __init__(self, host='0.0.0.0', port=9999):
        self.host = host
        self.port = port
        self.server_socket = None

        self.foundation_pose = None

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"[*] Server started on {self.host}:{self.port}")

        self._thread = threading.Thread(target=self._run, args=())
        self._thread.start()

    def _run(self):
        while True:
            print("[*] Waiting for a new connection...")
            try:
                conn, addr = self.server_socket.accept()
                print(f"[+] Client connected from {addr}")
                while True:
                    if not self._handle_client(conn):
                        break

                conn.close()
                print("[*] Connection closed")
            except Exception as e:
                print("[!] Accept failed:", e)

    def _handle_client(self, conn):
        try:
            buffer = b""
            while True:
                chunk = conn.recv(2 ** 24)
                if not chunk:
                    print("[*] Client disconnected")
                    break
                buffer += chunk
                if HEADER in buffer and FOOTER in buffer:
                    start = buffer.find(HEADER) + len(HEADER)
                    end = buffer.find(FOOTER)
                    payload = buffer[start:end]
                    break

            if not payload:
                print("[!] No valid payload found")
                return False

            size_len = struct.calcsize(SIZE_FORMAT)
            size = struct.unpack(SIZE_FORMAT, payload[:size_len])[0]
            data_bytes = payload[size_len:size_len + size]
            data_dict = pickle.loads(data_bytes)

            # 원하는 작업 수행
            action = data_dict["action"]
            print("Received keys:", data_dict.keys())
            print("Action:\n", action)

            if action == 'initialize':
                color = data_dict["color"]
                depth = data_dict["depth"]
                K = data_dict["K"]
                mask = data_dict["mask"]
                mesh_file = data_dict["mesh_file"]
                self.foundation_pose = FoundationPoseCore(color=color, depth=depth, mask=mask, K=K, mesh_file=mesh_file)
                response_dict = {
                    "status": "ok",
                    "message": "Data received successfully",
                }
            elif action == 'inference':
                color = data_dict["color"]
                depth = data_dict["depth"]
                K = data_dict["K"]
                T_pose, vis = self.foundation_pose.inference(color=color, depth=depth, K=K)

                # 응답 메시지 구성
                response_dict = {
                    "status": "ok",
                    "message": "Data received successfully",
                    "T_pose": T_pose,
                    "vis": vis,
                }
            else:
                # 응답 메시지 구성
                response_dict = {
                    "status": "error",
                    "message": "Unknown action",
                }
                print("[!] Unknown action")

            self._send_response(conn, response_dict)
            return True
        except Exception as e:
            print(f"[!] Error while handling client: {e}")
            return False

    def _send_response(self, conn, response_dict):
        serialized = pickle.dumps(response_dict)
        size = struct.pack(SIZE_FORMAT, len(serialized))
        payload = HEADER + size + serialized + FOOTER
        conn.sendall(payload)
        print("[*] Response sent to client")

class FoundationPoseTCPClient:
    def __init__(self, server_ip='127.0.0.1', port=9999):
        self.server_ip = server_ip
        self.port = port
        self.socket = None

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.server_ip, self.port))
        print(f"[+] Connected to {self.server_ip}:{self.port}")

    def send(self, data_dict, verbose=True):
        serialized = pickle.dumps(data_dict)
        size = struct.pack(SIZE_FORMAT, len(serialized))
        payload = HEADER + size + serialized + FOOTER
        self.socket.sendall(payload)
        if verbose:
            print("[+] Data sent")
        return self.receive_response()

    def close(self):
        if self.socket:
            self.socket.close()

    def receive_response(self):
        buffer = b""
        while True:
            chunk = self.socket.recv(4096)
            if not chunk:
                break
            buffer += chunk
            if HEADER in buffer and FOOTER in buffer:
                start = buffer.find(HEADER) + len(HEADER)
                end = buffer.find(FOOTER)
                payload = buffer[start:end]
                break

        size_len = struct.calcsize(SIZE_FORMAT)
        size = struct.unpack(SIZE_FORMAT, payload[:size_len])[0]
        data_bytes = payload[size_len:size_len + size]
        response = pickle.loads(data_bytes)
        return response

if __name__=='__main__':
    server = FoundationPoseTCPServer('localhost', 9999)
    server.start()

