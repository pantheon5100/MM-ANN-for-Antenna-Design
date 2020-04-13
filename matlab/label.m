categories = zeros(length(pr_all),1);
for i= 1:length(pr_all)
    data = pr_all{i,1};
    categories(i) = length(data);
end

plot(categories,".");
hist(categories);
