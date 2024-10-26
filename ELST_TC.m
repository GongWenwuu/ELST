function [Xest, U, hist] = ELST_TC(X,Omega,Opts) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                       Proximal Alternating Linearized Minimization for ELST                                    %
% \underset{\mathcal{G},\{\mathbf{U}_{n}\}}{\operatorname{min}} \ (1-\alpha) \prod_{n=1}^{N}\left\|\mathbf{U}_{n}\right\|_{*}
% + \alpha\|\mathcal{G}\|_{1} + \frac{\beta}{2}\left\|\mathcal{G} \times_{n=1}^{N} \mathbf{U}_{n} -\mathcal{X}^{0}\right\|_{\mathrm{F}}^{2}
%                                       This code was written by Wenwu Gong (2023.03)                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isfield(Opts, 'weight'); weight = 'lrstd';         else; weight = Opts.weight; end
if ~isfield(Opts, 'prior');  prior = 'lrstd';          else; prior = Opts.prior;   end
if ~isfield(Opts, 'flag');   flag = [];                else; flag = Opts.flag;     end
if ~isfield(Opts, 'lambda'); lambda = 1;               else; lambda = Opts.lambda; end
if isfield(Opts,'maxit');    maxit = Opts.maxit;     else; maxit = 500;        end
if isfield(Opts,'tol');      tol = Opts.tol;         else; tol = 1e-5;         end
if isfield(Opts,'phi');      phi = Opts.phi;         else; phi = 0;            end

N = ndims(X);
Z = X; 
% figure('Position',get(0,'ScreenSize'));
% % subplot(1,3,1);ImageShow3D(Z(:,:,5));title('Corrupted');
Z(~Omega) = mean(X(Omega)); 
[Ginit,Uinit,~] = Initial(Z,Opts.Rpara,Opts.init); 
[L, T, beta, flag] = LocalPrior(Opts.Xtr, Opts.Rpara, flag, prior);
 
Usq = cell(1,N); w_n = zeros(1,N); 
for n = 1:N
    Usq{n} = Uinit{n}'*Uinit{n};
    w_n(n) = MyWeight(Z, n, 'sum');
end
coeff = w_n;

t0 = 1;
Gextra = Ginit; Uextra = Uinit; U = Uinit; 
Lgnew = 1; LU0 = ones(N,1); LUnew = ones(N,1);
gradU = cell(N,1); wU = ones(N,1);

delta = 1.1; p = 0.8; q = 0.8; r = 4;
fprintf('Iteration:     ');
for iter = 1:maxit
    fprintf('\b\b\b\b\b%5i',iter);
    
    for n = 1:N
        % -- Core tensor updating --    
        gradG = lambda*gradientG(Gextra, U, Usq, Z); 
        Lg0 = Lgnew;
        Lgnew = lambda*lipG(Usq, delta);
        if Opts.alpha == 0
            G = Gextra - gradG/Lgnew;
        else
            G = thresholding(Gextra - gradG/Lgnew, Opts.alpha/Lgnew);
        end

        % -- Factor matrices updating --
        gradU{n} = gradientU(Uextra, U, Usq, G, Z, L{n}, T{n}, lambda, beta(n), n, flag(n));
        LU0(n) = LUnew(n);
        LUnew(n) = lipU(Usq, G, L{n}, T{n}, lambda, beta(n), n, flag(n), delta);
        if strcmp(weight,'lrstd')
            coeff(n) = MyWeight(Uextra, n, weight);
        elseif strcmp(weight,'tmac')
            obj = ModalProduct_All(G,U,'decompress');
            fit = TensorNorm(Omega.*(X-obj),'fro')^2;
            coeff(n) = 1./(fit.^2);
        end
        if Opts.alpha == 1
            U{n} = Uextra{n} - gradU{n}/LUnew(n);
        else
            [U{n}, ~, ~] = tracenorm(Uextra{n} - gradU{n}/LUnew(n), w_n(n)*(1-Opts.alpha)/LUnew(n));
        end

        Usq{n} = U{n}'*U{n};
    end

    w_n = coeff/sum(coeff);

    Z_pre = Z;
    GU = ModalProduct_All(G,U,'decompress');
    Z(~Omega) = GU(~Omega);
    Z(Omega) = X(Omega) + phi*(Z(Omega) - GU(Omega));

    % -- reporting --
    relchange = norm(Z(:)-Z_pre(:))/norm(Z_pre(:));
    hist.rel(iter) = relchange;
    if isfield(Opts,'Xtr')
        rmse = sqrt((1/length(nonzeros(~Omega)))*norm(Opts.Xtr(~Omega)-Z(~Omega),2)^2);
        hist.rmse(iter) = rmse;
        rse = norm(Opts.Xtr(~Omega)-Z(~Omega))/norm(Opts.Xtr(:));
        hist.rse(iter) = rse;
        nmae = norm(Opts.Xtr(~Omega)-Z(~Omega),1)/norm(Opts.Xtr(~Omega),1);
        hist.nmae(iter) = nmae;
    end
    
    % plot(hist.rse);axis([0,maxit,0,inf]);title('# iterations vs. RSEs');
    % % subplot(1,3,2);plot(hist.rse);axis([0,maxit,0,inf]);title('# iterations vs. RSEs');
    % % subplot(1,3,3);ImageShow3D(Z(:,:,5));title('Recovery');
    % % axes('position',[0,0,1,1],'visible','off');
    % pause(0.1);

    % -- extrapolation and stopping checks -- 
    t = (p+sqrt(q+r*t0^2))/2;
    w = (t0-1)/t;
    wG = min([w,0.9999*sqrt(Lg0/Lgnew)]);
    Gextra = G + wG*(G - Ginit); 
    for n = 1:N
        wU(n) = min([w,0.9999*sqrt(LU0(n)/LUnew(n))]);
        Uextra{n} = U{n}+wU(n)*(U{n}-Uinit{n});
    end
    Ginit = G; Uinit = U; t0 = t; 

    if relchange < tol
        break;
    end    

end

Xest = Z;
Xest(Omega) = X(Omega);

end