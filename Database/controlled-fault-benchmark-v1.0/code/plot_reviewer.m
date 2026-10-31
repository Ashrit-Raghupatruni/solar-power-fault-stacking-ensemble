function plot_reviewer()
% Ve 2 figure moi tu reviewer-fixes: fig10 disentangle, fig11 compression curve.
here=fileparts(mfilename('fullpath')); outdir=fullfile(here,'figs');
if ~exist(outdir,'dir'); mkdir(outdir); end
C.base=[0.35 0.40 0.47]; C.ext=[0.84 0.37 0.00]; C.gb=[0.00 0.45 0.70]; C.stat=[0.00 0.62 0.45]; C.freq=[0.80 0.47 0.65];
set(groot,'defaultAxesFontName','Arial','defaultTextFontName','Arial','defaultAxesFontSize',11,...
 'defaultAxesLineWidth',0.9,'defaultAxesBox','off','defaultAxesTickDir','out','defaultLegendBox','off');

%% FIG 10: disentangle — dong gop tung nhom feature (3 tram)
D=readtable(fullfile(here,'rev_disentangle.csv'),'TextType','string');
sets=["base7","base7+FFTwave","base7+freqdev","base7+stat","extended_all"];
setlab={'PE+stat (base)','+ FFT/wavelet','+ freq channel','+ multiscale stat','+ all (extended)'};
stns=["Station 1 (Lab)","Station 2 (Resid.)","Station 3 (Indust.)"]; stlab={'S1','S2','S3'};
M=nan(numel(sets),numel(stns));
for i=1:numel(sets); for j=1:numel(stns)
  r=D(D.Station==stns(j)&D.FeatureSet==sets(i),:); if ~isempty(r), M(i,j)=r.AUROC(1); end
end; end
fig=figure('Position',[80 80 860 460],'Color','w');
hb=bar(1:numel(sets),M,'grouped','EdgeColor','none','BarWidth',0.9);
cm=[C.gb;C.stat;C.freq]; for k=1:numel(hb); hb(k).FaceColor=cm(k,:); end
set(gca,'XTick',1:numel(sets),'XTickLabel',setlab,'XTickLabelRotation',18); ylim([0.6 0.95]);
ylabel('AUROC'); legend(stlab,'Location','northwest'); grid on; set(gca,'GridLineStyle',':','GridAlpha',0.25,'Layer','top');
title('Feature-group contribution (cumulative, per station)','FontWeight','bold','FontSize',12);
savef(fig,fullfile(outdir,'fig10_disentangle'));

%% FIG 11: compression size-accuracy
K=readtable(fullfile(here,'rev_compression.csv'),'TextType','string');
sz=double(K.size_KB); au=double(K.AUROC); ap=double(K.AUPRC); lab=cellstr(K.Config);
fig=figure('Position',[80 80 720 460],'Color','w');
semilogx(sz,au,'-o','Color',C.ext,'LineWidth',1.8,'MarkerFaceColor',C.ext,'MarkerSize',7); hold on;
semilogx(sz,ap,'-s','Color',C.gb,'LineWidth',1.8,'MarkerFaceColor',C.gb,'MarkerSize',7);
xl=xline(520,'--','ESP32 RAM (520 KB)'); xl.Color=[.45 .45 .45]; xl.FontSize=9;
for i=1:numel(sz); text(sz(i),au(i)+0.012,strrep(lab{i},'_','\_'),'FontSize',7.5,'HorizontalAlignment','center','Color',[.3 .3 .3]); end
xlabel('Model size (KB, log scale)'); ylabel('Score'); ylim([0.3 0.95]);
legend({'AUROC','AUPRC'},'Location','southwest'); grid on; set(gca,'GridLineStyle',':','GridAlpha',0.25,'Layer','top');
title('Compression: size vs accuracy (Station 1, extended)','FontWeight','bold','FontSize',12);
savef(fig,fullfile(outdir,'fig11_compression'));

%% FIG 12: threshold sensitivity (chong cherry-pick)
f=fullfile(here,'rev_threshold_sens.csv');
if isfile(f)
 try
  T=readtable(f,'TextType','string'); su=unique(T.Station,'stable'); cols=[C.gb;C.stat;C.freq];
  fig=figure('Position',[80 80 780 460],'Color','w'); hold on;
  for i=1:numel(su); s=T(T.Station==su(i),:); plot(s.threshold,s.EvF1,'-o','Color',cols(i,:),'LineWidth',1.8,'MarkerSize',4,'MarkerFaceColor',cols(i,:)); end
  xlabel('Decision threshold'); ylabel('Event-F1 (%)'); ylim([0 80]); xlim([0 1]);
  legend(cellstr(su),'Location','south','Orientation','horizontal'); grid on; set(gca,'GridLineStyle',':','GridAlpha',0.25,'Layer','top');
  title('Event-F1 vs threshold (broad plateau \rightarrow not cherry-picked)','FontWeight','bold','FontSize',12);
  savef(fig,fullfile(outdir,'fig12_threshold_sensitivity'));
 catch e; fprintf('[fig12 skip: %s]\n',e.message); end
end

%% FIG 13: feature extraction cost (ESP32 est)
f=fullfile(here,'rev_feature_cost.csv');
if isfile(f)
 try
  T=readtable(f,'TextType','string'); T=T(T.feature~="TOTAL",:);
  [~,o]=sort(T.esp32_est_us_per_window,'descend'); T=T(o,:); ms=T.esp32_est_us_per_window/1000;
  fig=figure('Position',[80 80 760 420],'Color','w');
  barh(categorical(strrep(T.feature,"_"," "),flipud(strrep(T.feature,"_"," "))),ms,'FaceColor',C.base,'EdgeColor','none');
  xlabel('Estimated ESP32 time per sample (ms)'); grid on; set(gca,'GridLineStyle',':','GridAlpha',0.25,'Layer','top');
  title('Feature-extraction cost (est. ESP32 @240MHz); total \approx5 ms \ll 1 s','FontWeight','bold','FontSize',11);
  savef(fig,fullfile(outdir,'fig13_feature_cost'));
 catch e; fprintf('[fig13 skip: %s]\n',e.message); end
end

%% FIG 14: fault-type x day coverage (chong leakage)
f=fullfile(here,'rev_faulttype_x_day.csv');
if isfile(f)
 try
  T=readtable(f,'TextType','string','ReadRowNames',true);
  M=[double(T.St1) double(T.St2) double(T.St3)]; types=string(T.Properties.RowNames);
  fig=figure('Position',[80 80 620 640],'Color','w');
  imagesc(M); colormap(flipud(bone)); clim([0 7]); cb=colorbar; cb.Label.String='# distinct days (of 7)';
  set(gca,'XTick',1:3,'XTickLabel',{'St1','St2','St3'},'YTick',1:numel(types),'YTickLabel',strrep(types,"_"," "),'TickDir','out');
  for r=1:numel(types); for c=1:3; tc=[0 0 0]; if M(r,c)<=2; tc=[0.8 0.2 0]; end; text(c,r,sprintf('%d',M(r,c)),'HorizontalAlignment','center','FontSize',9,'Color',tc); end; end
  title('Fault type \times day coverage (\geq2 = leakage-safe)','FontWeight','bold','FontSize',11);
  savef(fig,fullfile(outdir,'fig14_faulttype_x_day'));
 catch e; fprintf('[fig14 skip: %s]\n',e.message); end
end
fprintf('DONE reviewer figs\n');
end

function savef(fig,base)
ok={};
try, exportgraphics(fig,[base '.png'],'Resolution',300); ok{end+1}='png'; catch e, fprintf('[png %s]\n',e.message); end
okp=false; try, exportgraphics(fig,[base '.pdf'],'ContentType','vector'); okp=true; catch, end
if ~okp, try, set(fig,'Units','inches'); p=get(fig,'Position'); set(fig,'PaperPositionMode','manual','PaperUnits','inches','PaperPosition',[0 0 p(3) p(4)],'PaperSize',[p(3) p(4)]); print(fig,[base '.pdf'],'-dpdf','-vector','-r300'); okp=true; catch e, fprintf('[pdf %s]\n',e.message); end; end
if okp, ok{end+1}='pdf'; end
fprintf('  saved %s (%s)\n',base,strjoin(ok,'/'));
end
