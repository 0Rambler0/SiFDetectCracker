% parpool(38);
read_path = '/home/dslab/hx/vfuzz/media/tmp/15/';
save_path = '/home/dslab/hx/vfuzz/media/hx_workspace/SiFDetectCracker/refactor_project/target/Deep4SNet/audio_img/15/';
;Path = strcat(read_path);
File = dir(fullfile(Path, '*.wav'));
Filename = {File.name};
len = length(File);
nbins = 65536; % number of bins of the histogram.
histFcn = @(x, y) histogram(x, y);

parfor i = 1:len
    [voice, FS] = audioread(strcat(Path, File(i).name));
    h = histFcn(voice, nbins);
    name = File(i).name(1:end - 4);
    saveas(h, strcat(save_path, name), 'jpg');
    if i*20 == 0
        fprintf('已处理图片数量:%d\n',i);
    end
end
exit();
