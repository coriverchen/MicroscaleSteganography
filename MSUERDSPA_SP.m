function MSUERDSPA_SP(payload,dirSource,output_dir)


warning off
clc
currentFolder = pwd;
addpath(genpath(currentFolder))
% addpath('..\ml_stc');
% addpath('..\jpegtbx_1.4');
% payload = 0.3;
% QFs = 75:20:95;wetConst
wetConst = 10^13;
QFs = 75;
Q =75;
%CAPA = 20:10:50;         %payload 0.1-0.5
scheme = 'J-UERD';
MSG_SEED = 518;
SEED = 438;
H = 10;      % constraint height - default is 10 - drives the complexity/quality tradeof

    
    dirSource_qf = dirSource;%[dirSource '\Q' num2str(QF)];  % input dir for each QF
    Output_path = output_dir;%[output_dir '\Q' num2str(QF) '\' scheme '\' num2str(CAPACITY)];   % output dir of each scheme
    if ~exist(Output_path,'dir'); mkdir(Output_path); end
    % if exist(outdir,'dir'); rmdir(outdir,'s'); end    
    % if ~exist(outdir,'dir'); mkdir(outdir); end
    files=dir([dirSource_qf '/*.jpg']);
    
    pc = parcluster('local');
    % explicitly set the JobStorageLocation to the temp directory that was
    % created in your sbatch script
    pc.JobStorageLocation = strcat('/public/chenkj/', getenv('SLURM_JOB_ID'));
    % start the parallel pool with 12 workers
    j = str2num(getenv('SLURM_CPUS_PER_TASK'));
    fprintf('parpool size: %d\n', j);
    parpool(j);
    
    parfor w=1:length(files)        
        if mod(w,1000)==0 
            fprintf('%3d, processing %s \n',w,files(w).name); 
        end

        if files(w).isdir==0
            full_image_file_name=[dirSource_qf '/' files(w).name];
            stego_name =  [Output_path '/' files(w).name];
            
            img = jpeg_read(full_image_file_name); 
            img1 = imread(full_image_file_name);
            UM_IMG=UMA_fun(img1,1);
            temp = load(strcat('default_gray_jpeg_obj_', num2str(Q), '.mat'));
            default_gray_jpeg_obj = temp.default_gray_jpeg_obj;
            C_STRUCT = default_gray_jpeg_obj;
            C_QUANT = C_STRUCT.quant_tables{1};


            fun=@dct2;
            xi= blkproc(double(UM_IMG)-128,[8 8],fun);
            % Quantization
            fun = @(x) x./C_QUANT;
            DCT_real = blkproc(xi,[8 8],fun);
            DCT_rounded = round(DCT_real);       
            
            dct_coef = img.coef_arrays{1};
            UM_dct_coef = DCT_real;%%  important
            
            
            dct_coef_cover = dct_coef;
            [img_h, img_w]  = size(dct_coef);
            cover_size = img_h * img_w;
            dct_coef2 = dct_coef;
            % remove DC coefs;
            dct_coef2(1:8:end,1:8:end) = 0;
            UM_dct_coef(1:8:end,1:8:end) = 0;
            

            nz_index = find(dct_coef < 10000000); % use all dct coefficients
            nz_number = nnz(dct_coef2); % number of non zero ac coefficients
            
            %hidden_message = double(rand(ceil(max(CAPACITY)*nz_number/100+1),1)>0.5);
             
            q_tab = img.quant_tables{1};
            q_tab(1,1) = 0.5*(q_tab(2,1)+q_tab(1,2));
            q_matrix = repmat(q_tab,[64 64]);
            
            %%% energy of each block
%             fun = @(block_struct) sum(sum(abs(q_tab.*block_struct.data)))*ones(8);
%             J = blockproc(dct_coef2,[8 8],fun);
            UM_dct_coef = im2col(q_matrix.*UM_dct_coef,[8 8],'distinct');
            J2M = sum(abs(UM_dct_coef));
            JUM = ones(64,1)*J2M;
            JUM = col2im(JUM,[8 8], [512 512], 'distinct');
            
            % dct_coef2 = im2col(q_matrix.*dct_coef2,[8 8],'distinct');
           % J2 = sum(abs(dct_coef2));
            % J = ones(64,1)*J2;
            % J = col2im(J,[8 8], [512 512], 'distinct');
            J = JUM;
            
            
%             decide = q_matrix./J; % version 1

            pad_size = 8;
            im2 = padarray(J,[pad_size pad_size],'symmetric'); % energies of eight-neighbor blocks
            size2 = 2*pad_size;
            im_l8 = im2(1+pad_size:end-pad_size,1:end-size2);
            im_r8 = im2(1+pad_size:end-pad_size,1+size2:end);
            im_u8 = im2(1:end-size2,1+pad_size:end-pad_size);
            im_d8 = im2(1+size2:end,1+pad_size:end-pad_size);
            im_l88 = im2(1:end-size2,1:end-size2);
            im_r88 = im2(1+size2:end,1+size2:end);
            im_u88 = im2(1:end-size2,1+size2:end);
            im_d88 = im2(1+size2:end,1:end-size2);
            JJ = J+  (0.25*(im_l8+im_r8+im_u8+im_d8)+0.25*(im_l88+im_r88+im_u88+im_d88));
            decide = q_matrix./abs(JJ); % version J+ 2   
            decide = decide/min(decide(:));



            rho = decide;

            for i = 1:8
                for j = 1:8
                    L1 = fspecial('average',[3 3]);
                    rho(i:8:end,j:8:end)= conv2(rho(i:8:end,j:8:end), L1, 'same');
                end
            end
            rhoP1 = rho;
            rhoM1 = rho;
%             
%           decide = decide(nz_index(r_index));
%           decide = decide/min(decide);
            
%           rhoP1 = decide;
%           rhoM1 = decide;
                    
%             costs = zeros(3, length(nz_index), 'single'); % for each pixel, assign cost of being changed
%             costs(1,:) = decide;       % cost of changing the first cover pixel by -1, 0, +1
%             costs(3,:) = decide;       % cost of changing the first cover pixel by -1, 0, +1
            rhoP1(rhoP1 > wetConst) = wetConst;
            rhoP1(isnan(rhoP1)) = wetConst;    
            rhoP1(dct_coef_cover > 1023) = wetConst;
    
            rhoM1(rhoM1 > wetConst) = wetConst;
            rhoM1(isnan(rhoM1)) = wetConst;
            rhoM1(dct_coef_cover < -1023) = wetConst;
            
            nzAC = nnz(dct_coef_cover)-nnz(dct_coef_cover(1:8:end,1:8:end));
            stego = simulate_embed(dct_coef_cover, rhoP1, rhoM1, round(payload*nzAC));
            
            %[d,stego,n_msg_bits,l] = stc_pm1_pls_embed(int32(nz_dct_coef)', costs, uint8(hidden_message)', H); % ternary STC embedding 
            % extr_msg = stc_ml_extract(stego, n_msg_bits, H);
            % sum(uint8(hidden_message)'~=extr_msg);
            
            %em_dct_coef = dct_coef;
            %em_dct_coef(nz_index(r_index)) =stego;
        
            temp = load(strcat('default_gray_jpeg_obj_', num2str(QFs), '.mat'));
            default_gray_jpeg_obj = temp.default_gray_jpeg_obj;
            C_STRUCT = default_gray_jpeg_obj;
            C_STRUCT.coef_arrays{1} = stego;

            jpeg_write(C_STRUCT, stego_name);
%             img.coef_arrays{1} = em_dct_coef;
%             jpeg_write(img,stego_name);                   % generate stego image     
     
        end
    end
          poolobj = gcp('nocreate');
      delete(poolobj);
end
% end end
