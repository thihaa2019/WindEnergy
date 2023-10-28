function experiments()
    rep = 10;
    valuation = zeros(rep,3);
    for i =1:rep
        [valuation(i,1), ~, ~] = microgrid_gp2d();
    end

%     for i =1:rep
%         [valuation(i,2), ~, ~] =  microgrid_Cagedgp2();
%     end

    
    for i =1:rep
        [valuation(i,3), ~, ~] =  microgrid_Adaptivegp2d();
    end

    header = {'SOBOL', 'TD', 'Adaptive'};
    result = [header; num2cell(valuation); num2cell(mean(valuation)); num2cell(std(valuation))];
    fid = fopen('microgridGR2D.csv', 'w') ;
    fprintf(fid, '%s,', result{1,1:end-1}) ;
    fprintf(fid, '%s\n', result{1,end}) ;
    fclose(fid) ;
    dlmwrite('microgridGR2D.csv', result(2:end,:), '-append') ;

    
    
end
