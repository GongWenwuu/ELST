dense_tensor = MSI;
MSI_Eval_ELST_PALM = zeros(9,13);
DLP = [0.1:0.1:0.9,0.93,0.95,0.97,0.99];
for MR = 1:length(DLP)
    rng('default')
    filename=['MSI_ELST_' num2str(MR) '.mat'];
    sample_ratio = 1- DLP(MR);
    sample_num = round(sample_ratio*numel(dense_tensor));
    fprintf('Sampling tensor with %4.1f%% known elements ...... \n',100*sample_ratio);
    % Filter missing positions 
    idx = 1:numel(dense_tensor);
    idx = idx(dense_tensor(:)>0);
    % Artificial missing position
    mask = sort(randperm(length(idx),sample_num));
    arti_miss_idx = idx;  
    arti_miss_idx(mask) = [];  
    arti_miss_mv = dense_tensor(arti_miss_idx);
    Omega = zeros(size(dense_tensor)); Omega(mask) = 1; Omega = boolean(Omega);
    sparse_tensor = Omega.*dense_tensor;
    fprintf('Known elements / total elements: %6d/%6d.\n',sample_num,numel(dense_tensor));
    clear idx 

    t0 = tic;
    Opts = Initial_Para(500,size(dense_tensor),'lrstd',0.5); Opts.Xtr = dense_tensor; Opts.prior = 'lrstd'; Opts.flag = [1,1,1]; 
    [est_tensor, ~, PALM_info] = ELST_TC(sparse_tensor,Omega,Opts); 
    MSI_Eval_ELST_PALM(9,MR) = toc(t0);
    save(filename,"Omega",'sparse_tensor','est_tensor','dense_tensor',"PALM_info","Opts")

    rse = TensorNorm(est_tensor - dense_tensor,'fro')/TensorNorm(dense_tensor,'fro');
    nmae = norm(arti_miss_mv-est_tensor(arti_miss_idx),1) / norm(arti_miss_mv,1);
    rmse = sqrt((1/length(arti_miss_mv))*norm(arti_miss_mv-est_tensor(arti_miss_idx),2)^2);  
    [psnr, ssim, fsim, ergas, msam] = MSIQA(dense_tensor, est_tensor);
    MSI_Eval_ELST_PALM(1,MR) = psnr; MSI_Eval_ELST_PALM(2,MR) = ssim; MSI_Eval_ELST_PALM(3,MR) = fsim; MSI_Eval_ELST_PALM(4,MR) = ergas; 
    MSI_Eval_ELST_PALM(5,MR) = msam; MSI_Eval_ELST_PALM(6,MR) = rmse; MSI_Eval_ELST_PALM(7,MR) = nmae; MSI_Eval_ELST_PALM(8,MR) = rse;

end

t0 = tic;
Opts = Initial_Para(300,size(dense_tensor),'lrstd',0.5); Opts.Xtr = dense_tensor; Opts.prior = 'teo'; Opts.flag = [1,1,1];
[est_tensor, ~, PALM_info] = ELST_TC(sparse_tensor,Omega,Opts);
toc(t0)
[psnr, ssim, fsim, ergas, msam] = MSIQA(dense_tensor, est_tensor)

t0 = tic;
[est_tensor, ~, PLADMM_info] = PLADMM_ELST_TC(sparse_tensor,Omega,Opts);
toc(t0)
