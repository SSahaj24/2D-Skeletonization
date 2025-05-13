function data = load_np(filename)
% Load numpy array into Matlab
%
% data = load_np(filename)
%
% Basically converts the numpy file to a temporary mat file using
% Python.exe, followed by loading the mat file
%
% The function assumes Python.exe is on the Windows path
% with SciPy and NumPy both installed. 
%
% D.Kroon 12-10-2021

filename_mat = fullfile(tempdir,'temp.mat');
if(isfile(filename_mat))
    delete(filename_mat)
end

python_script = fullfile(tempdir,'numpy2mat.py');
if(~isfile(python_script))
    fid = fopen(python_script,'w');
    fprintf(fid,'%s\n','import scipy.io as sio');
    fprintf(fid,'%s\n','import numpy as np');
    fprintf(fid,'%s\n','import sys');
    fprintf(fid,'%s\n','data = np.load(sys.argv[1], allow_pickle=True)');
    fprintf(fid,'%s\n','m=dict()');
    fprintf(fid,'%s\n','m["data"]=data');
    fprintf(fid,'%s\n','sio.savemat(sys.argv[2],m)');
    fclose(fid);
end

c = ['python.exe "',  python_script,'" "',filename, '" "', filename_mat,'"'];
system(c);
data =load(filename_mat);
data = data.data;
%data = permute(data,ndims(data):-1:1);
