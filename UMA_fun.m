function R = UMA_fun(I_raw,alpha)
%unsharp
%������Ĥ
%ǿ����������
%2016.5.5
%by �¿ɽ�

%%
H1 =1/((1+2*alpha)^2) *[alpha^2 -alpha-2*alpha^2 alpha^2;
-alpha-2*alpha^2 (1+2*alpha)^2 -alpha-2*alpha^2;
alpha^2 -alpha-2*alpha^2 alpha^2;
];


% H1 = [0.25 -1 0.25;
%       -1 4 -1;
%       0.25 -1 0.25];

%H1 = [-1 2 -1; 2 -3 2;-1 2 -1];


% compute residual
R = conv2(I_raw, H1, 'same');

