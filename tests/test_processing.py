import numpy as np
from src.data_loader import SignalRecord
from src.preprocessing import preprocess_signal, window_record
from src.feature_extraction import extract_signal_features

def test_preprocessing_repairs_and_standardizes():
    x=np.array([1,2,np.nan,4,np.inf,6,7,8.]);y=preprocess_signal(x,100,{"detrend":False,"normalization":"standard"});assert np.isfinite(y).all();assert abs(y.mean())<1e-10

def test_windowing_metadata():
    r=SignalRecord(signal=np.arange(100.),sampling_rate=10,scenario="1",sensor="7",source_file="x")
    w=window_record(r,{"size_seconds":2,"overlap":.5},{"detrend":False,"normalization":"none"});assert len(w)==9;assert w[0].end_sample==20

def test_feature_extraction_zero_safe():
    f=extract_signal_features(np.zeros(128),100);assert f["crest_factor"]==0;assert all(np.isfinite(list(f.values())))
