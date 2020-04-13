%% Data Extraction and Conversion
S11_val=responses{36,1}; % S11_val the response of interest (specify it accordingly)

complex_S11_temp=S11_val(:,2:3); % select the complex values of S11 in S11_val
complex_S11=complex_S11_temp(:,1)+complex_S11_temp(:,2).*1i; % convert the selected values from double to complex
S11_abs=abs(complex_S11); % derive the magnitude or absolute value of the complex values

S11_dB = 20*log10(S11_abs); % convert to decibels 

%% Plot

omega=4.5e9:0.00199990000000039e9:6.5e9; % Not the bounds and step size in the frequency points
wf=omega/1e9; % frequency unit in GHz

plot(wf,S11_dB,'r','LineWidth', 2.5);
xlabel('Freq. in GHz')
ylabel ('S_1_1 in dB')
grid on

