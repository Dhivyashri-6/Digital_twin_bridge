import zipfile
import numpy as np
from scipy.io import savemat
from src.data_loader import load_aaa_file, load_mat_file

def test_aaa_z24_header(tmp_path):
    p=tmp_path/"s.aaa";p.write_text("s.aaa|8Averages\n4\n0.01\n1\n2\n3\n4\n")
    data,meta=load_aaa_file(p);assert np.allclose(data,[1,2,3,4]);assert meta["sample_interval"]==.01

def test_aaa_malformed_lines(tmp_path):
    p=tmp_path/"s.aaa";p.write_text("header\n1, 2\nbad\n3, 4\n")
    data,_=load_aaa_file(p);assert data.shape==(2,2)

def test_mat_loader(tmp_path):
    p=tmp_path/"s.mat";savemat(p,{"acc":np.arange(12).reshape(6,2)})
    result=load_mat_file(p);assert result["acc"].shape==(6,2)
