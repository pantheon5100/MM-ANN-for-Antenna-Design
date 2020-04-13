%% Rational Function Approxixmation of S parameter
clear -global
close all
mode = "test";
if mode == "test"
    load("./data/Test_Data.mat");
    generate_rp = test_responses;
else
    load("./data/Training_Data.mat");
    generate_rp = responses;
end
pr_all = cell(length(generate_rp), 6);
checker = cell(length(generate_rp), 1);
for i =(1:length(generate_rp))
    freq = generate_rp{i,1}(:,1);
    % frequency scaling and shifting
     freq = 0.01*freq + 10;

%     data = complex(generate_rp{i,1}(:,2), generate_rp{i,1}(:,3));
    data = generate_rp{i,1}(:,2)+ generate_rp{i,1}(:,3).*1i;

    % Fit a rational function to the data using rationalfit. 
    %RATIONALFIT Perform rational fitting to complex frequency-dependent data.
    %   FIT = RATIONALFIT(FREQ,DATA) uses vector fitting with complex
    %   frequencies S = j*2*pi*FREQ to construct a rational function fit
    %
    %            C(1)     C(2)           C(n)
    %   F(S) =  ------ + ------ + ... + ------
    %           S-A(1)   S-A(2)         S-A(n)
    %
    %  C -> Residue  A -> Pole
    
    fit_data = rationalfit(freq,data, 'Tolerance', -40);
%     fit_data = rationalfit(freq,data, -45, 'NPoles', [10 14]);
    
    ar = real(fit_data.A);
    ai = imag(fit_data.A);
    cr = real(fit_data.C);
    ci = imag(fit_data.C);
    % sort the data
    [ai, I] = sort(ai);
    ar = ar(I);
    cr = cr(I);
    ci = ci(I);
    
%     index = ai>=0;
%     pr_all{i,1} = ar(index);
%     pr_all{i,2} = ai(index);
%     pr_all{i,3} = cr(index);
%     pr_all{i,4} = ci(index);

    pr_all{i,1} = ar;
    pr_all{i,2} = ai;
    pr_all{i,3} = cr;
    pr_all{i,4} = ci;

    
    % Compute the frequency response of the rational function using freqresp. 
    [resp, f] = freqresp(fit_data,freq);
    
    % Compute origin meap
    observed = data;
    predicted = resp;
    meap = mean(abs((observed - predicted)./observed))*100;
    pr_all{i,5} = meap;
    pr_all{i,6} = length(pr_all{i,4});
    
    disp(["Data: ", num2str(i), " MAPE: ", num2str(meap), " Order:", num2str(length(pr_all{i,1})), num2str(length(pr_all{i,2})), num2str(length(pr_all{i,3})), num2str(length(pr_all{i,4}))]);
end

Index = [pr_all{:,6}];
for order=Index
    pr_ac = [pr_all{Index==order, 1};pr_all{Index==order, 2};pr_all{Index==order, 3};pr_all{Index==order, 4}]';
    save("./data4/"+mode+"_pr"+num2str(order), "pr_ac");
end
save("./data4/" +mode+ "_index", "Index");

figure
plot([pr_all{:,6}],".");

%% plot freq response compare
freq = generate_rp{i,1}(:,1);
data = complex(generate_rp{i,1}(:,2), generate_rp{i,1}(:,3));
% resp = complex(pr_all{i,1}, pr_all{i,2});
figure
% plot(freq/1e9,20*log10(abs(data)), 'LineWidth', 2.5)
plot(freq/1e9,20*log10(abs(data)), 'LineWidth', 2.5)
hold on
% plot(freq/1e9,20*log10(abs(resp)),'r--', 'LineWidth', 2.5);
plot(freq/1e9,20*log10(abs(resp)),'r--', 'LineWidth', 2.5);
legend(["Oringin", "Fit"]);
title('Rational fitting of S11 magnitude')
figure
plot(freq/1e9,unwrap(angle(data)), 'LineWidth', 2.5)
hold on
plot(freq/1e9,unwrap(angle(resp)),'r--', 'LineWidth', 2.5)
legend(["Oringin", "Fit"]);
title('Rational fitting of S11 angle')
