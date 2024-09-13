'''
Get 3d semantics onto dslr images by rasterizing the mesh
and append depth image from render
'''

import argparse
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import torch

from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    fisheyecameras
)

from common.scene_release import ScannetppScene_Release
from common.file_io import load_json, load_yaml_munch, read_txt_list
from semantic.utils.resizing_function_for_label_images import resize_labels_majority_voting
from semantic.utils.undistortion import undistort_image
from semantic.utils.colmap_utils import read_cameras_text, read_images_text, camera_to_intrinsic
# from render import
import os
import sys
from pathlib import Path

import imageio
try:
    import renderpy
except ImportError:
    print("renderpy not installed. Please install renderpy from https://github.com/liu115/renderpy")
    sys.exit(1)

from common.utils.colmap import read_model, write_model, Image
from common.scene_release import ScannetppScene_Release
from common.utils.utils import run_command, load_yaml_munch, load_json, read_txt_list

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print('Device:', device)

def main(args):

    # read cfg
    cfg = load_yaml_munch(args.config_file)
    output_dir = cfg.get("output_dir")
    resize_width = cfg.get("resize_width")
    resize_height = cfg.get("resize_height")
    if output_dir is None:
        # default to data folder in data_root
        output_dir = Path(cfg.data_root) / "data"
    output_dir = Path(output_dir)

    scene_list = read_txt_list(cfg.scene_list)

    pth_data_dir = Path(cfg.pth_data_dir)

    for scene_id in tqdm(scene_list, desc='scene'):
        scene = ScannetppScene_Release(scene_id, data_root=cfg.data_root)
        render_engine = renderpy.Render()
        render_engine.setupMesh(str(scene.scan_mesh_path))
        cameras, images, points3D = read_model(scene.dslr_colmap_dir, ".txt")
        camera = next(iter(cameras.values()))
        fx, fy, cx, cy = camera.params[:4]
        params = camera.params[4:]
        camera_model = camera.model
        render_engine.setupCamera(
            camera.height, camera.width,
            fx, fy, cx, cy,
            camera_model,
            params,  # Distortion parameters np.array([k1, k2, k3, k4]) or np.array([k1, k2, p1, p2])
        )
        near = cfg.get("near", 0.05)
        far = cfg.get("far", 20.0)
        # read mesh
        mesh = o3d.io.read_triangle_mesh(str(scene.scan_mesh_path))
        # read annotation
        pth_data = torch.load(pth_data_dir / f'{scene_id}.pth')

        # list of dslr images
        dslr_names_all = load_json(scene.dslr_train_test_lists_path)['train']
        # pick every nth dslr image and corresponding camera pose
        dslr_indices = list(range(0, len(dslr_names_all), cfg.dslr_subsample_factor))
        dslr_names = [dslr_names_all[i] for i in dslr_indices]

        # read camera intrinsics
        intrinsics_file = scene.dslr_colmap_dir / 'cameras.txt'
        # there is only 1 camera model, get it
        colmap_camera = list(read_cameras_text(intrinsics_file).values())[0]
        # params [0,1,2,3] give the intrinsic
        intrinsic_mat = camera_to_intrinsic(colmap_camera)
        # rest are the distortion params
        # need 6 radial params
        distort_params = list(colmap_camera.params[4:]) + [0, 0]

        extrinsics_file = scene.dslr_colmap_dir / 'images.txt'
        all_extrinsics = read_images_text(extrinsics_file)
        # get the extrinsics for the selected images into a dict with filename as key
        all_extrinsics_dict = {v.name: v.to_transform_mat() for v in all_extrinsics.values()}

        # create meshes object
        verts = torch.Tensor(np.array(mesh.vertices))
        faces = torch.Tensor(np.array(mesh.triangles))
        meshes = Meshes(verts=[verts.to(device)], faces=[faces.to(device)])

        # EDITED: Initialize separate containers for each component
        image_names = []
        original_images = []
        semantic_labels = []
        depth_images = []
        camera_params = []

        # go through dslr images
        for _, image_name in enumerate(tqdm(dslr_names, desc='image')):
            for image_id, image in images.items():
                if image.name != image_name:
                    continue
                world_to_camera = image.world_to_camera
                _, depth, _ = render_engine.renderAll(world_to_camera, near, far)
                depth = depth/(1168/448) # resize depth to
                # Make depth in mm and clip to fit 16-bit image
                depth = (depth.astype(np.float32) * 1000).clip(0, 65535).astype(np.uint16)
                break
            # draw the camera frustum on the mesh
            camera_pose = all_extrinsics_dict[image_name]

            # read image and get dims
            image_path = scene.dslr_resized_dir / image_name
            image = plt.imread(image_path)
            # get h, w from image
            img_height, img_width = image.shape[:2]

            raster_settings = RasterizationSettings(image_size=(img_height, img_width),
                                                blur_radius=0.0,
                                                faces_per_pixel=1,
                                                cull_to_frustum=True)
            rasterizer = MeshRasterizer(
                raster_settings=raster_settings
            )

            # get 2d-3d mapping of this image by rasterizing, add a dimension in the beginning
            R = torch.Tensor(camera_pose[:3, :3]).unsqueeze(0)
            T = torch.Tensor(camera_pose[:3, 3]).unsqueeze(0)

            # create camera with opencv function
            image_size = torch.Tensor((img_height, img_width))
            image_size_repeat = torch.tile(image_size.reshape(-1, 2), (1, 1))
            intrinsic_repeat = torch.Tensor(intrinsic_mat).unsqueeze(0).expand(1, -1, -1)

            opencv_cameras = cameras_from_opencv_projection(
                # N, 3, 3
                R=R,
                # N, 3
                tvec=T,
                # N, 3, 3
                camera_matrix=intrinsic_repeat,
                # N, 2 h,w
                image_size=image_size_repeat
            )

            # apply the same transformation for fisheye cameras
            # transpose R, then negate 1st and 2nd columns
            fisheye_R = R.mT
            fisheye_R[:, :, :2] *= -1
            # negate x and y in the transformation T
            # negate everything
            fisheye_T = -T
            # negate z back
            fisheye_T[:, -1] *= -1

            # focal, center, radial_params, R, T, use_radial
            fisheye_cameras = fisheyecameras.FishEyeCameras(
                focal_length=opencv_cameras.focal_length,
                principal_point=opencv_cameras.principal_point,
                radial_params=torch.Tensor([distort_params]),
                use_radial=True,
                R=fisheye_R,
                T=fisheye_T,
                image_size=image_size_repeat,
                # need to specify world coordinates, otherwise camera coordinates
                world_coordinates=True
            )

            # rasterize
            with torch.no_grad():
                raster_out = rasterizer(meshes, cameras=fisheye_cameras.to(device))
                # H, W
                pix_to_face = raster_out.pix_to_face.squeeze().cpu().numpy()

            valid_pix_to_face = pix_to_face[:, :] != -1
            mesh_faces_np = np.array(mesh.triangles)

            pix_inst_ids = np.zeros_like(pix_to_face)
            # get instance ids on pixels
            pix_inst_ids[valid_pix_to_face] = pth_data['vtx_instance_anno_id'][mesh_faces_np[pix_to_face[valid_pix_to_face]][:, 0]]
            # get semantic labels on pixels, initialize to -1
            pix_sem_ids = np.ones_like(pix_to_face) * -1
            # get semantic labels on pixels
            pix_sem_ids[valid_pix_to_face] = pth_data['vtx_labels'][mesh_faces_np[pix_to_face[valid_pix_to_face]][:, 0]]
            semantic_label = pix_sem_ids.astype(np.int16)
            # label_max = np.max(semantic_label)
            # label_min = np.min(semantic_label)

            # undisort the image

            semantic_label = undistort_image(semantic_label, intrinsic_mat, distort_params, True)
            image = undistort_image(image, intrinsic_mat, distort_params, False)
            depth = undistort_image(depth, intrinsic_mat, distort_params, False)
            semantic_label = semantic_label.astype(np.int16)

            # resize all data to 672*448
            # linear interpolation for image and depth downscaling
            # getting the most frequently occurring label in the resized area for semantic labels
            image = cv2.resize(image, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
            semantic_label = resize_labels_majority_voting(semantic_label, (resize_width, resize_height))

            # EDITED: Collect data for the current image in separate containers
            image_names.append(image_name)
            original_images.append(image)
            semantic_labels.append(semantic_label)
            depth_images.append(depth)
            camera_params.append({
                'focal_length': opencv_cameras.focal_length,
                'principal_point': opencv_cameras.principal_point,
                'radial_params': distort_params,
                'R': fisheye_R,
                'T': fisheye_T,
                'use_radial': True,
                'intrinsic_mat': intrinsic_mat,
                'extrinsic': camera_pose,
                'view_width_px': resize_width,
                'view_height_px': resize_height
            })

        # EDITED: Save the collected data for the entire scene after processing all images
        pth_data_dir_2d = Path(output_dir)
        save_path = pth_data_dir_2d / f'{scene_id}.pth'
        torch.save({
            'scene_id': scene_id,
            'image_name': image_names,
            'original_image': original_images,
            '2d_semantic_labels': semantic_labels,
            'depth_image': depth_images,
            'camera_params': camera_params
        }, save_path)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    config_file_path = '/home/hua/Desktop/ml3d/scannetpp/semantic/configs/rasterize_render.yml'
    parser = argparse.ArgumentParser(description='Process 3D and 2D data.')
    parser.add_argument('config_file', type=str, nargs='?', default=config_file_path,
                        help='Path to the configuration file.')
    args = parser.parse_args()
    main(args)