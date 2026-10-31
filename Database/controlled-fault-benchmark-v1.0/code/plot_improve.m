function plot_improve()
% plot_improve.m — Publication-quality figures (real controlled-fault data)
% Doc: ket_qua_cai_thien.csv, ket_qua_per_type.csv, ket_qua_mutual_info.csv (neu co),
%      ket_qua_dl.csv (neu co). Xuat figs/*.png (300 DPI) + *.pdf + *.eps (vector).
% Chuan: Okabe-Ito colorblind-safe, Arial, truc sach, nhan gia tri gon.
% Chay: matlab -batch "cd('<...>/_IMPROVE'); plot_improve"

here = fileparts(mfilename('fullpath'));
fmain = fullfile(here,'ket_qua_cai_thien.csv');
fpt   = fullfile(here,'ket_qua_per_type.csv');
fmi   = fullfile(here,'ket_qua_mutual_info.csv');
fdl   = fullfile(here,'ket_qua_dl.csv');
fmcp  = fullfile(here,'ket_qua_multiclass_perclass.csv');
fmccm = fullfile(here,'ket_qua_multiclass_confusion.csv');
fadap = fullfile(here,'ket_qua_adaptive.csv');
fadapp= fullfile(here,'ket_qua_adaptive_pertype.csv');
fabl  = fullfile(here,'ket_qua_rpd_ablation.csv');
outdir= fullfile(here,'figs'); if ~exist(outdir,'dir'); mkdir(outdir); end
assert(isfile(fmain),'Thieu %s',fmain);

% ---- Okabe-Ito colorblind-safe palette ----
C.base = [0.35 0.40 0.47];   % slate (baseline)
C.ext  = [0.84 0.37 0.00];   % vermillion (extended/improved)
C.gb   = [0.00 0.45 0.70];   % blue
C.xgb  = [0.00 0.62 0.45];   % bluish green
C.lgb  = [0.80 0.47 0.65];   % reddish purple
C.ent  = [0.60 0.60 0.60];   % entropy (gray)
C.stat = [0.00 0.45 0.70];   % statistical (blue)
C.spec = [0.84 0.37 0.00];   % spectral (orange)
C.prop = [0.84 0.37 0.00];   % proposed
setpub();

T = readtable(fmain,'TextType','string');
T.Stn = "S"+extractBetween(T.Station,"Station "," ");
fB="baseline 7-feat"; fE="extended +spectral";
stns = unique(T.Stn,'stable');

%% ===== FIG 1: baseline vs extended (GB) — AUROC & EvF1 =====
gb = T(T.Model=="GB",:);
auB=getv(gb,stns,fB,'AUROC'); auE=getv(gb,stns,fE,'AUROC');
f1B=getv(gb,stns,fB,'EvF1');  f1E=getv(gb,stns,fE,'EvF1');
fig=figure('Position',[80 80 1040 430],'Color','w');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
nexttile; grouped([auB auE],stns,{'Baseline (PE+RPD)','+ Spectral'},[C.base;C.ext],'AUROC','%.3f');
ylim([0 1.05]); yline(0.5,'--','random','Color',[.45 .45 .45],'FontSize',8,'LabelHorizontalAlignment','left');
panel('(a) AUROC');
nexttile; grouped([f1B f1E],stns,{'Baseline (PE+RPD)','+ Spectral'},[C.base;C.ext],'Event-F1 (%)','%.1f');
ylim([0 max(100,max([f1B;f1E])*1.18)]); panel('(b) Event-F1');
sgt('Effect of spectral features on real controlled-fault data (Gradient Boosting)');
savef(fig,fullfile(outdir,'fig1_baseline_vs_extended'));

%% ===== FIG 2: per-type AUROC (centerpiece) =====
if isfile(fpt)
    P=readtable(fpt,'TextType','string'); pg=P(P.Model=="GB",:);
    if ~isempty(pg)
        gB=grpmean(pg(pg.FeatureSet==fB,:)); gE=grpmean(pg(pg.FeatureSet==fE,:));
        ty=unique([gB.type;gE.type],'stable');
        aB=nan(numel(ty),1); aE=nan(numel(ty),1);
        for i=1:numel(ty)
            j=gB.type==ty(i); if any(j),aB(i)=gB.AUROC(j);end
            k=gE.type==ty(i); if any(k),aE(i)=gE.AUROC(k);end
        end
        [~,o]=sort(aE-aB,'ascend','MissingPlacement','first'); ty=ty(o);aB=aB(o);aE=aE(o);
        lbl=strrep(ty,"_"," ");
        fig=figure('Position',[80 80 780 max(440,38*numel(ty))],'Color','w');
        hb=barh(categorical(lbl,lbl),[aB aE],'BarWidth',0.85);
        hb(1).FaceColor=C.base; hb(2).FaceColor=C.ext;
        hb(1).EdgeColor='none'; hb(2).EdgeColor='none';
        xlim([0 1.0]); xline(0.5,'--','random','Color',[.45 .45 .45],'FontSize',8,'LabelVerticalAlignment','bottom');
        xlabel('Per-type AUROC'); grid on; set(gca,'GridLineStyle',':','GridAlpha',0.25,'Layer','top');
        legend({'Baseline (PE+RPD)','+ Spectral'},'Location','southoutside','Orientation','horizontal');
        panel('Per-anomaly-type AUROC (GB, mean over stations)');
        savef(fig,fullfile(outdir,'fig2_per_type_auroc'));
    end
end

%% ===== FIG 3: GB vs XGB vs LGB (extended) =====
mdls=unique(T.Model,'stable');
if numel(mdls)>1
    ext=T(T.FeatureSet==fE,:); cmap=[C.gb;C.xgb;C.lgb];
    auc=nan(numel(stns),numel(mdls)); f1=nan(numel(stns),numel(mdls));
    for j=1:numel(mdls)
        auc(:,j)=getv(ext(ext.Model==mdls(j),:),stns,fE,'AUROC');
        f1(:,j)=getv(ext(ext.Model==mdls(j),:),stns,fE,'EvF1');
    end
    cmap=cmap(1:numel(mdls),:);
    fig=figure('Position',[80 80 1040 430],'Color','w');
    tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
    nexttile; grouped(auc,stns,cellstr(mdls),cmap,'AUROC','%.3f'); ylim([0 1.05]);
    yline(0.5,'--','Color',[.45 .45 .45]); panel('(a) AUROC by classifier');
    nexttile; grouped(f1,stns,cellstr(mdls),cmap,'Event-F1 (%)','%.1f');
    ylim([0 max(100,max(f1(:))*1.18)]); panel('(b) Event-F1 by classifier');
    sgt('Classifier comparison on extended features');
    savef(fig,fullfile(outdir,'fig3_classifier_comparison'));
end

%% ===== FIG 4: Mutual Information (neu co) =====
if isfile(fmi)
    M=readtable(fmi,'TextType','string');
    if any(strcmp(M.Properties.VariableNames,'MI_mean'))
        M=sortrows(M,'MI_mean','descend');
        cols=zeros(height(M),3);
        for i=1:height(M)
            switch string(M.domain(i))
                case "entropy", cols(i,:)=C.ent;
                case "statistical", cols(i,:)=C.stat;
                case "spectral", cols(i,:)=C.spec;
                otherwise, cols(i,:)=[.5 .5 .5];
            end
        end
        lbl=strrep(M.feature,"_"," ");
        fig=figure('Position',[80 80 720 max(420,34*height(M))],'Color','w');
        hb=barh(categorical(lbl,flipud(lbl)),M.MI_mean,'FaceColor','flat','EdgeColor','none');
        hb.CData=cols;
        xlabel('Mutual information with label (mean over stations)');
        grid on; set(gca,'GridLineStyle',':','GridAlpha',0.25,'Layer','top');
        % legend domain
        hold on;
        h1=barh(nan,nan,'FaceColor',C.stat,'EdgeColor','none');
        h2=barh(nan,nan,'FaceColor',C.spec,'EdgeColor','none');
        h3=barh(nan,nan,'FaceColor',C.ent,'EdgeColor','none');
        legend([h1 h2 h3],{'statistical','spectral','entropy'},'Location','southeast');
        panel('Per-feature mutual information (real-fault data)');
        savef(fig,fullfile(outdir,'fig4_mutual_info'));
    end
end

%% ===== FIG 5: DL baselines vs proposed (neu co) =====
if isfile(fdl)
    D=readtable(fdl,'TextType','string'); D.Stn="S"+extractBetween(D.Station,"Station "," ");
    dmdls=unique(D.Model,'stable');
    propF1=getv(T(T.Model=="GB",:),stns,fE,'EvF1');   % proposed = PE+RPD+GB extended
    M=nan(numel(stns),numel(dmdls)+1);
    for j=1:numel(dmdls)
        for i=1:numel(stns)
            r=D(D.Stn==stns(i)&D.Model==dmdls(j),:); if ~isempty(r),M(i,j)=r.EvF1(1);end
        end
    end
    M(:,end)=propF1;
    leg=[cellstr(dmdls);{'PE+RPD+GB (proposed)'}];
    cmap=[repmat([.55 .60 .66],numel(dmdls),1); C.prop];
    fig=figure('Position',[80 80 760 440],'Color','w');
    grouped(M,stns,leg,cmap,'Event-F1 (%)','%.1f'); ylim([0 max(100,max(M(:),[],'omitnan')*1.18)]);
    panel('Engineered features vs raw-window deep learning');
    savef(fig,fullfile(outdir,'fig5_dl_vs_proposed'));
end

%% ===== FIG 6: multi-class confusion matrix (extended, primary LGB) =====
if isfile(fmccm)
 try
  CM=readtable(fmccm,'TextType','string'); mp="LGB"; if ~any(CM.Model==mp), mp=CM.Model(1); end
  Cs=CM(CM.Model==mp,:); cats=unique([Cs.true;Cs.pred]); K=numel(cats); Mm=zeros(K,K);
  for i=1:height(Cs); r=find(cats==Cs.true(i)); c=find(cats==Cs.pred(i)); Mm(r,c)=Mm(r,c)+Cs.count(i); end
  Mn=Mm./max(sum(Mm,2),1);
  fig=figure('Position',[80 80 720 660],'Color','w');
  imagesc(Mn); colormap(flipud(gray)); clim([0 1]); cb=colorbar; cb.Label.String='Row-normalized (recall)';
  set(gca,'XTick',1:K,'XTickLabel',cats,'YTick',1:K,'YTickLabel',cats,'XTickLabelRotation',40,'TickDir','out');
  xlabel('Predicted category'); ylabel('True category'); axis square;
  for r=1:K; for c=1:K; if Mn(r,c)>=0.005
      tc=[0.1 0.1 0.1]; if Mn(r,c)>0.5; tc=[1 1 1]; end
      text(c,r,sprintf('%.2f',Mn(r,c)),'HorizontalAlignment','center','FontSize',8,'Color',tc);
  end; end; end
  panel(sprintf('Fault-identification confusion matrix (%s, extended)',mp));
  savef(fig,fullfile(outdir,'fig6_confusion_matrix'));
 catch e; fprintf('  [fig6 skip: %s]\n',e.message); end
end

%% ===== FIG 7: per-category F1 (multi-class, baseline vs extended) =====
if isfile(fmcp)
 try
  PC=readtable(fmcp,'TextType','string'); mp="LGB"; if ~any(PC.Model==mp), mp=PC.Model(1); end
  P=PC(PC.Model==mp,:); cats=unique(P.category); f1b=nan(numel(cats),1); f1e=nan(numel(cats),1);
  for i=1:numel(cats)
    b=P(P.category==cats(i)&P.FeatureSet=="baseline 7-feat",:); if ~isempty(b),f1b(i)=mean(b.f1);end
    e=P(P.category==cats(i)&P.FeatureSet=="extended +spectral",:); if ~isempty(e),f1e(i)=mean(e.f1);end
  end
  [~,o]=sort(f1e,'ascend','MissingPlacement','first'); cats=cats(o);f1b=f1b(o);f1e=f1e(o);
  fig=figure('Position',[80 80 720 max(420,42*numel(cats))],'Color','w');
  hb=barh(categorical(cats,cats),[f1b f1e],'BarWidth',0.82);
  hb(1).FaceColor=C.base; hb(2).FaceColor=C.ext; hb(1).EdgeColor='none'; hb(2).EdgeColor='none';
  xlim([0 1]); xlabel('Per-category F1 (mean over stations)'); grid on;
  set(gca,'GridLineStyle',':','GridAlpha',0.25,'Layer','top');
  legend({'Baseline (PE+RPD)','+ Spectral'},'Location','southoutside','Orientation','horizontal');
  panel(sprintf('Multi-class fault identification: per-category F1 (%s)',mp));
  savef(fig,fullfile(outdir,'fig7_per_category_f1'));
 catch e; fprintf('  [fig7 skip: %s]\n',e.message); end
end

%% ===== FIG 8: adaptive routing vs baseline/global =====
if isfile(fadap)
 try
  A=readtable(fadap,'TextType','string'); A.Stn="S"+extractBetween(A.Station,"Station "," ");
  stnsA=unique(A.Stn,'stable'); meths=["baseline-7feat","global-extended","adaptive-routing"];
  Mx=nan(numel(stnsA),numel(meths));
  for i=1:numel(stnsA); for j=1:numel(meths)
    r=A(A.Stn==stnsA(i)&A.Method==meths(j),:); if ~isempty(r),Mx(i,j)=r.AUROC(1);end
  end; end
  fig=figure('Position',[80 80 1060 440],'Color','w');
  tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
  nexttile; grouped(Mx,stnsA,{'Baseline','Global-extended','Adaptive routing'},[C.base;[.45 .55 .66];C.ext],'AUROC','%.3f');
  ylim([0 1.05]); panel('(a) AUROC by fusion strategy');
  nexttile;
  if isfile(fadapp)
    AP=readtable(fadapp,'TextType','string');
    gg=grpmean(AP(AP.Method=="global-extended",:)); aa=grpmean(AP(AP.Method=="adaptive-routing",:));
    ty=unique([gg.type;aa.type],'stable'); vg=nan(numel(ty),1); va=nan(numel(ty),1);
    for i=1:numel(ty)
      j=gg.type==ty(i); if any(j),vg(i)=gg.AUROC(j);end
      k=aa.type==ty(i); if any(k),va(i)=aa.AUROC(k);end
    end
    [~,o]=sort(va-vg,'descend','MissingPlacement','last'); ty=ty(o);vg=vg(o);va=va(o);
    sh=min(8,numel(ty)); lbl=strrep(ty(1:sh),"_"," ");
    hb=barh(categorical(lbl,flipud(lbl)),[vg(1:sh) va(1:sh)],'BarWidth',0.82);
    hb(1).FaceColor=[.45 .55 .66]; hb(2).FaceColor=C.ext; hb(1).EdgeColor='none'; hb(2).EdgeColor='none';
    xlim([0 1]); xlabel('Per-type AUROC'); grid on; set(gca,'GridLineStyle',':','GridAlpha',0.25);
    legend({'Global-ext','Adaptive'},'Location','southoutside','Orientation','horizontal');
    panel('(b) Types most recovered by routing');
  end
  sgt('Adaptive feature routing vs naive concatenation');
  savef(fig,fullfile(outdir,'fig8_adaptive_routing'));
 catch e; fprintf('  [fig8 skip: %s]\n',e.message); end
end

%% ===== FIG 9: RPD ablation (+RPD vs -RPD ~ bang nhau -> RPD du thua) =====
if isfile(fabl)
 try
  AB=readtable(fabl,'TextType','string'); AB.Stn="S"+extractBetween(AB.Station,"Station "," ");
  stnsB=unique(AB.Stn,'stable');
  % chi xet 'extended' (cau hinh chinh): +RPD vs -RPD, metric AUROC
  wp=nan(numel(stnsB),1); wo=nan(numel(stnsB),1);
  for i=1:numel(stnsB)
    a=AB(AB.Stn==stnsB(i)&AB.FeatureSet=="extended+RPD",:); if ~isempty(a),wp(i)=a.AUROC(1);end
    b=AB(AB.Stn==stnsB(i)&AB.FeatureSet=="extended-RPD",:); if ~isempty(b),wo(i)=b.AUROC(1);end
  end
  fig=figure('Position',[80 80 640 440],'Color','w');
  grouped([wp wo],stnsB,{'With RPD','Without RPD'},[C.base;C.spec],'AUROC (extended)','%.3f');
  ylim([0 1.05]); yline(0.5,'--','Color',[.45 .45 .45]);
  panel('RPD ablation: removing RPD does not hurt (RPD is redundant)');
  savef(fig,fullfile(outdir,'fig9_rpd_ablation'));
 catch e; fprintf('  [fig9 skip: %s]\n',e.message); end
end

fprintf('\nDONE. Figures -> %s\n',outdir);
end

% ================= helpers =================
function setpub()
set(groot,'defaultAxesFontName','Arial','defaultTextFontName','Arial',...
 'defaultAxesFontSize',11,'defaultAxesLineWidth',0.9,'defaultAxesBox','off',...
 'defaultAxesTickDir','out','defaultAxesTickLength',[0.012 0.012],...
 'defaultAxesXColor',[0.15 0.15 0.15],'defaultAxesYColor',[0.15 0.15 0.15],...
 'defaultLegendBox','off','defaultFigureColor','w');
end
function v=getv(Tsub,stns,fs,col)
v=nan(numel(stns),1);
for i=1:numel(stns)
    r=Tsub(Tsub.Stn==stns(i)&Tsub.FeatureSet==fs,:); if ~isempty(r),v(i)=r.(col)(1);end
end
end
function grouped(Mx,xlab,leg,cmap,ylab,fmt)
ng=size(Mx,1);
hb=bar(1:ng,Mx,'grouped','EdgeColor','none','BarWidth',0.9);  % numeric x (vector-safe)
for k=1:numel(hb); hb(k).FaceColor=cmap(k,:); end
set(gca,'XTick',1:ng,'XTickLabel',cellstr(xlab),'XLim',[0.4 ng+0.6]);
ylabel(ylab); legend(leg,'Location','best','AutoUpdate','off'); grid on;
set(gca,'GridLineStyle',':','GridAlpha',0.25,'Layer','top');
for k=1:numel(hb)
    text(hb(k).XEndPoints,hb(k).YEndPoints,compose(fmt,Mx(:,k)'),...
        'HorizontalAlignment','center','VerticalAlignment','bottom',...
        'FontSize',7.5,'Color',[.25 .25 .25]);
end
end
function g=grpmean(Q)
[u,~,ic]=unique(Q.anomaly_type,'stable'); m=accumarray(ic,Q.AUROC,[],@mean);
g=table(u,m,'VariableNames',{'type','AUROC'});
end
function panel(t); title(t,'FontWeight','bold','FontSize',12); end
function sgt(t); sgtitle(t,'FontWeight','bold','FontSize',13,'FontName','Arial'); end
function savef(fig,base)
ok={};
% PNG (raster) qua exportgraphics
try, exportgraphics(fig,[base '.png'],'Resolution',300); ok{end+1}='png'; catch e, fprintf('  [png fail: %s]\n',e.message); end
% PDF vector: thu exportgraphics, fallback sang print (engine on dinh hon cho colorbar/CData)
okpdf=false;
try, exportgraphics(fig,[base '.pdf'],'ContentType','vector'); okpdf=true; catch, end
if ~okpdf
    try
        set(fig,'Units','inches'); p=get(fig,'Position'); set(fig,'PaperPositionMode','manual','PaperUnits','inches','PaperPosition',[0 0 p(3) p(4)],'PaperSize',[p(3) p(4)]);
        print(fig,[base '.pdf'],'-dpdf','-vector','-r300'); okpdf=true;
    catch e, fprintf('  [pdf fallback fail: %s]\n',e.message); end
end
if okpdf, ok{end+1}='pdf'; end
% EPS vector: thu exportgraphics, fallback print
okeps=false;
try, exportgraphics(fig,[base '.eps'],'ContentType','vector'); okeps=true; catch, end
if ~okeps
    try, print(fig,[base '.eps'],'-depsc','-vector','-r300'); okeps=true; catch e, fprintf('  [eps fallback fail: %s]\n',e.message); end
end
if okeps, ok{end+1}='eps'; end
fprintf('  saved %s (%s)\n',base,strjoin(ok,'/'));
end
