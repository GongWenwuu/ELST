function [Xest, U, hist] = PLADMM_ELST_TC(X,Omega,Opts) 

if isfield(Opts,'maxit');   maxit = Opts.maxit;  else; maxit = 500;     end
if isfield(Opts,'mu');      mu = Opts.mu;        else; mu = 1;          end
if isfield(Opts,'tol');     tol = Opts.tol;      else; tol = 1e-5;      end

N = ndims(X);
Z = X; 
% figure('Position',get(0,'ScreenSize'));
Z(~Omega) = mean(X(Omega));
[G,U,~] = Initial(Z,Opts.Rpara,Opts.init); 
Y = zeros(size(X));
O = zeros(size(X));

[Lap, Teo, beta, flag] = LocalPrior(Opts.Xtr, Opts.Rpara, Opts.flag, 'lrstd');
coeff = zeros(1,N); 
for n = 1:N
    coeff(n) = MyWeight(Z, n, 'sum');
end

mu_rho = 1.15; 
sigma = cell(1,N); Xnorm = zeros(1,N);
for n = 1:N
    Xn  = reshape(permute(Z, [n 1:n-1 n+1:N]),size(Z,n),[]);
    sigma{n} = svd(Xn, 'econ');
    Xnorm(n)  = sum(sigma{n}>eps);
end
mu = mu*sum(circshift(Xnorm, [1,1]).*circshift(Xnorm, [2,2]));

muY = mu;
muO = mu;

fprintf('Iteration:     ');
for iter = 1:maxit
    fprintf('\b\b\b\b\b%5i',iter);

    % update G
    [G, ~] = apgG(G,U,Z,Y,Opts.alpha,muY);

    % update U
    for n = 1:N
        [U{n}, ~] = apgU(U,G,Z,Y,Opts.alpha,coeff(n),flag(n),beta(n),Lap{n},Teo{n},muY,n);
        coeff(n) = MyWeight(U, n, 'lrstd');
    end
    coeff = coeff/sum(coeff);

    Z_pre = Z;
    GU = ModalProduct_All(G,U,'decompress');
    
    % update Z
    Z(~Omega) = (muY*GU(~Omega)-Y(~Omega))/muY; 
    Z(Omega) = (muY*GU(Omega)-Y(Omega) + muO*X(Omega)-O(Omega))/(muY+muO); 

    % update multiplayers
    Y = Y+muY*(Z-GU);
    O(Omega) = O(Omega)+muO*(Z(Omega)-X(Omega)); 
    muY = muY*mu_rho;
    muO = muO*mu_rho;

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
    % pause(0.1);

    % -- stopping checks --
    if relchange < tol 
        break;
    end
    
end

Xest = Z;
Xest(Omega) = X(Omega);

end