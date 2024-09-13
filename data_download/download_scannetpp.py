
'''
Download ScanNet++ data

Default: download splits with scene IDs and default files
that can be used for novel view synthesis on DSLR and iPhone images
and semantic tasks on the mesh
'''

import argparse
from pathlib import Path
from urllib.request import urlretrieve
import urllib.error
import yaml
from munch import Munch
from tqdm import tqdm
import json
import zipfile

from scene_release import ScannetppScene_Release

def read_txt_list(path):
    with open(path) as f: 
        lines = f.read().splitlines()

    return lines

def load_json(path):
    with open(path) as f:
        j = json.load(f)

    return j

def load_yaml_munch(path):
    with open(path) as f:
        y = yaml.load(f, Loader=yaml.Loader)

    return Munch.fromDict(y)

def download_file(url, filename, verbose=True, make_parent=False):
    '''
    Download file from url to filename
    '''
    # download_url = url.

    if make_parent:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f'{url} ==> {filename}')
    try:
        urlretrieve(url, filename)
        return True
    except urllib.error.HTTPError:
        print(f'ERROR: File not found: {url}')
        return False

def check_download_file(cfg, remote_path, local_path):
    '''
    check if file exists, else download 
    remote_path: relative to root
    local_path: full path
    '''
    if local_path.is_file():
        if cfg.verbose:
            print('File exists, skipping download: ', local_path)
        return True
    else:
        url = cfg.root_url.replace('TOKEN', cfg.token).replace('FILEPATH', str(remote_path))
        return download_file(url, local_path, verbose=cfg.verbose, make_parent=True)

def main(args):
    cfg = load_yaml_munch(args.config_file)

    missing = []

    data_root = Path(cfg.data_root)

    # create data root directory
    data_root.mkdir(parents=True, exist_ok=True)

    # download meta files
    for path in cfg.meta_files:
        if not check_download_file(cfg, path, data_root / path):
            missing.append(data_root / path)

    if cfg.metadata_only:
        print('Downloaded metadata, done.')
        return
    
    # read all the split files
    split_lists = {}
    for split in cfg.splits:
        split_path = data_root / 'splits' / f'{split}.txt'
        split_lists[split] = read_txt_list(split_path)

    # get the list of scenes to be downloaded
    if cfg.get('download_scenes'):
        scene_ids = cfg.download_scenes
    elif cfg.get('download_splits'):
        scene_ids = []
        for split in cfg.download_splits:
            split_path = Path(cfg.data_root) / 'splits' / f'{split}.txt'
            scene_ids += read_txt_list(split_path)

    # get the list of assets to download for these scenes
    if cfg.get('download_assets'):
        download_assets = cfg.download_assets
    elif cfg.get('download_options'):
        download_assets = []
        for option in cfg.download_options:
            option_assets = cfg.option_assets[option]
            for asset in option_assets:
                if asset not in download_assets:
                    download_assets.append(asset)
    else:
        download_assets = cfg.default_assets

    print('Downloading assets:', download_assets)
    print(f'Scenes selected: ', len(scene_ids))

    for scene_id in tqdm(scene_ids, desc='scenes'):
        # download from here
        # path relative to root url
        src_scene = ScannetppScene_Release(scene_id, data_root='data')
        # to here
        tgt_scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root) / 'data')

        # get the split for this scene
        split = None
        for split in cfg.splits:
            if scene_id in split_lists[split]:
                break
        assert split is not None, f'Scene {scene_id} not in any split'

        for asset in tqdm(download_assets, desc='assets', leave=False):
            # some assets not present in test splits
            if asset in cfg.exclude_assets.get(split, []):
                continue

            # check if asset is zipped, download the zip and unzip it to the target path
            if asset in cfg.zipped_assets:
                tgt_path = getattr(tgt_scene, asset)
                
                if tgt_path.is_file() or tgt_path.is_dir():
                    if cfg.verbose:
                        print('File exists, skipping download: ', tgt_path)
                    continue

                src_download_path = getattr(src_scene, asset).with_suffix('.zip')
                tgt_download_path = tgt_path.with_suffix('.zip')
                
                if not check_download_file(cfg, src_download_path, tgt_download_path):
                    missing.append(tgt_download_path)

                # unzip it
                if cfg.verbose:
                    print('Unzipping:', tgt_download_path)
                with zipfile.ZipFile(tgt_download_path, 'r') as zip_ref:
                    zip_ref.extractall(tgt_download_path.parent)
                # remove the zip file
                if cfg.verbose:
                    print('Delete zip file:', tgt_download_path)
                tgt_download_path.unlink()
            else:
                #  download single file
                src_path = getattr(src_scene, asset)
                tgt_path = getattr(tgt_scene, asset)
                if not check_download_file(cfg, src_path, tgt_path):
                    missing.append(tgt_path)

    if missing:
        print(f'{len(missing)} files missing:', missing)
    else:
        print('Download successful!')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('config_file', help='Path to config file')
    args = p.parse_args()

    main(args)