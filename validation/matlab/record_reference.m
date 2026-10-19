function record_reference(matlab_repo, case_name, Ndim, seed, outdir)
% Run the ORIGINAL MATLAB code (one SeMIS run followed by one SuS run, as in
% one iteration of drive_SuS_SeMIS.m) and record every random number drawn.
%
%   record_reference('/path/to/Sequential-multiple-importance-sampling-for-multimodal-Bayesian-inference', 'loggamma', 2, 1, '../reference')
%
% case_name : 'shells' | 'loggamma' | 'eggbox'
% The functions randn/rand/randperm/randi/randsample in this folder shadow the
% built-in ones and log every draw; rec_*.mat holds the stream and res_*.mat
% the results. validate.py replays the stream through the Python port.
here = fileparts(mfilename('fullpath'));
addpath(genpath(matlab_repo));
addpath(here);                         % shims first on the path
if exist('OCTAVE_VERSION', 'builtin')
    addpath(fullfile(here, 'octave'));  % Octave compatibility shims, and fminbnd with MATLAB's default tolerances
    pkg load statistics
    builtin('rand', 'seed', seed); builtin('randn', 'seed', seed);
else
    rng(seed);
end
switch case_name
    case 'shells',   Tfun = LKF_nD_GaussianShells(Ndim);
    case 'loggamma', Tfun = LKF_nD_GaussianLogGamma(Ndim);
    case 'eggbox',   Tfun = LKF_2D_Eggbox;
end
rec_reset();
IntBay_SuS   = SuS_Bay_Nataf(Tfun);
IntBay_SeMIS = SeMIS_Bay_Nataf(Tfun);
smc_SuS   = SeMCS(ES(IntBay_SuS));
smc_SeMIS = SeMCS(ES(IntBay_SeMIS));
smc_SeMIS.NumSam = 1700;
smc_SuS.NumSam   = 1000;

out_sm = smc_SeMIS.RunIte;
out_mis = MIS(out_sm);
z_mis = out_mis.Z;
[u_pos_semis, ~] = out_mis.PosSim;
z_sis = out_mis.C(end) + out_sm.g{end}(2);
G_semis = cell2mat(out_sm.g); Ncal_semis = out_sm.Ncal; Nite_semis = out_sm.Nite;

out_sm = smc_SuS.RunIte;
out_sus = IntBay_SuS.BayInf(out_sm);
z_sus = out_sus.z; u_pos_sus = out_sus.x_pos;
G_sus = cell2mat(out_sm.g); Ncal_sus = out_sm.Ncal; Nite_sus = out_sm.Nite; %#ok<NASGU>

tag = sprintf('%s_%d_%d', case_name, Ndim, seed);
rec_save(fullfile(outdir, ['rec_' tag '.mat']));
save(fullfile(outdir, ['res_' tag '.mat']), '-v7', 'z_mis', 'z_sis', 'z_sus', 'u_pos_semis', 'u_pos_sus', ...
     'G_semis', 'G_sus', 'Ncal_semis', 'Ncal_sus', 'Nite_semis', 'Nite_sus');
end
