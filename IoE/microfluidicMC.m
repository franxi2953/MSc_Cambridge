clear all; close all;
%% Smoldyn simulation parameters
simDuration = 20; % (s)
simTimeStep = 0.005; % (s)
samplingStartTime = 15; % should be large enough to eliminate the initial time transients (s)
numSamples = 1000; % number of samples
samplingPeriod = 0.005; % (s)


% Microfluidic channel parameters
h_ch = 5e-6; % channel height (m)
w_ch = 10e-6; % channel width (m)
A_ch = h_ch*w_ch; % cross-sectional area of the channel (m^2)
u = 10e-6; % average flow velocity (m/s)
x_r = 1e-3; % receiver's center position (m)
t_d = (x_r/u); % arrival time for peak (s)

% Molecule parameters
D_0 = 2e-11; % intrinsic diffusion coefficient (m^2/s)
D = (1+((8.5*u^2*h_ch^2*w_ch^2)/(210*D_0^2*(h_ch^2+2.4*h_ch*w_ch+w_ch^2))))*D_0; % effective diffusion coefficient (m^2/s)
K_b_m = 2e-17; % binding rate of ligands (m^3/s)
K_u_m = 1; % unbinding rate of ligands (1/s)
K_D_m = K_u_m/K_b_m; % dissociation constant of ligands (1/m^3)


% Receiver parameters
N_r = 200;  % number of independent receptors
l_gr = 5e-6; % length of the graphene channel along the microfluidic channel (m) corresponding to "width" of graphene in transistor
w_gr = w_ch; % width of the graphene channel across the microfluidic channel (m) corresponding to "length" of graphene in transistor
A_gr = l_gr * w_gr; % area of graphene surface exposed to electrolyte (μm)^2


% geometry of the microfluidic channel 
w_ch = w_ch * 1e6; % channel width (um)
h_ch = h_ch * 1e6; % channel height (um)
l_ch = 200;  % channel length - required only for Smoldyn simulations (um)
vol_ch = w_ch * h_ch * l_ch; % volume of the microfluidic channel (um^3)

% position for 'channel' compartment definition (required only for Smoldyn)
x_comp = round(l_ch/2); % (um)
y_comp = round(h_ch/2); % (um)
z_comp = round(w_ch/2); % (um)

%%%%%%%%%%%%modulation and bitstream generation%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
N_1 = 1000;
N_2 = 200;
Nt = [N_2 N_1]; % number of transmitted molecules for bit [0 1]

numSymbol = 1;
bitstream = randi([0,1],numSymbol,1);
N_m_array = zeros(numSymbol, 1);
N_m_array(bitstream == 0) = Nt(1);
N_m_array(bitstream == 1) = Nt(2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Ligand concentration samples (at peak-concentration sampling time)
c_m_peak_array = N_m_array / (A_ch*sqrt(4*pi*D*t_d));
% concentrations into numbers for Smoldyn
cLigand_array = c_m_peak_array * 1e-18; % (1/um^3)
numLigand_array = round(cLigand_array * vol_ch);
numReceptor = N_r;

% geometry of the receiver
l_rx = l_gr * 1e6; % length of receiver  along the microfluidic axis (um)
w_rx = w_gr * 1e6; % width of receiver across the microfluidic axis (um)
x_rx = round(l_ch/2); % receiver's center position (um), different than time-domain analysis, specify only for Smoldyn simulations

% diffusion coefficients of molecules
diffLigand = round(D*1e12); % effective diffusion constant (um^2/s)

% reaction rates
kBindLigand = K_b_m*1e18; % (um^3/s)
kUnbindLigand = K_u_m; % (1/s)


%%%%% SMOLDYN %%%%%%%%%%%%%%%%%%
%sample parameter check 
if samplingStartTime > simDuration 
    disp('Error: increase the duration of the simulation!')
    return
elseif mod(samplingPeriod,simTimeStep) ~= 0 
    disp('Error: enter a new samplingPeriod that is integer multiple of simTimeStep!')
    return
else
    samplingWindowLength = numSamples * samplingPeriod;
    if samplingStartTime + samplingWindowLength > simDuration
        disp('Error: Sampling extends beyond the simulation duration, decrease number of samples, sampling period, sampling start time, OR increase simulation duration.')
        return
    end
    samplingWindowLengthinIndex = samplingWindowLength / simTimeStep;
    samplingPeriodinIndex = samplingPeriod / simTimeStep;
    fprintf('Starting simulations... \n Sampling Window Length = %d seconds \n Sampling Frequency = %d Hz \n', samplingWindowLength, 1/samplingPeriod)
end

parfor i = 1:numSymbol
command = sprintf('smoldyn config.txt -wt --define simDuration=%d --define simTimeStep=%d --define numLigand=%d --define numReceptor=%d --define diffLigand=%d --define kBindLigand=%d  --define kUnbindLigand=%d --define w_ch=%d --define h_ch=%d --define l_ch=%d --define l_rx=%d --define w_rx=%d --define x_rx=%d --define x_comp=%d --define y_comp=%d --define z_comp=%d --define index=%d',simDuration, simTimeStep, numLigand_array(i),  numReceptor, diffLigand, kBindLigand, kUnbindLigand, w_ch, h_ch, l_ch, l_rx, w_rx, x_rx, x_comp, y_comp, z_comp, i);
fprintf(command);
system(command); 
end
%%
%%oneSample = zeros(numSymbol, 1);

for i = 1:numSymbol   
outputfilename = sprintf('configout_symbol_id_%d_moleculecount.txt', i);
molcount = importdata(outputfilename,' ');


samplingStartIndex = round(samplingStartTime / simTimeStep); 

molcount = molcount(samplingStartIndex : samplingPeriodinIndex : samplingStartIndex + samplingWindowLengthinIndex -1,:);

sample_boundreceptors = molcount(:,4);
time = molcount(:,1);

%%
% write a code for taking one sample from the peak approximately middle for



end
figure; semilogx(time,sample_boundreceptors); xlabel('time(s)'); ylabel('N_B');

%%%%%DETECTION THRESHOLD %%%%%%%%%%%%%%%%%
% delta_I_mean = zeros(2,1);
% delta_I_var = zeros(2,1);
% for i = [0 1]
%     N_m = Nt(i+1);
%     c_m = N_m/(A_ch*sqrt(4*pi*D*t_d));
%     
%     P_B_exp = (c_m/K_D_m)/(1 + c_m/K_D_m); % bound state probability of a receptor in the absence of interference
%
%    
%
%     delta_I_var(i+1) = .... ;
% end
% lambdaTH = thresholdCSK(delta_I_var(2),delta_I_var(1), delta_I_mean(2),delta_I_mean(1));
%%%%%%%%%%%%%%%%%%%%%%%

%%%DETECTION %%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%BEP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

