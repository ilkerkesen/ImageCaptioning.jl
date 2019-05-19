function train(; name="exp1", hiddensize=512, embedsize=512, batchsize=64,
               seed=-1, datadir=joinpath(DIR, "../data"), epochs=5,
               vggfile=joinpath(datadir, "imagenet-vgg-verydeep-16.mat"),
               atype=Sloth._atype)
    seed != -1 && Knet.seed!(seed)
    data = Dataset(datadir)
    vocab = build_vocab(data)
    model = ShowAndTell(; hiddensize=hiddensize, embedsize=embedsize,
                        vggfile=vggfile, vocabsize=length(vocab.w2i),
                        atype=atype)
    dtrn = TrainLoader(vocab, data.trn, data.images_path, data.features_path;
                       batchsize=batchsize, atype=atype)
    dval = EvalLoader(vocab, data.val, data.images_path, data.features_path;
                      atype=atype)
    f = (img,txt,_)->model(img,txt)
    bestscore = 0.0
    modelpath = abspath(joinpath(DIR, "../models"))
    savefile1 = joinpath(modelpath, "$(name)-best.jld2")
    savefile2 = joinpath(modelpath, "$(name)-curr.jld2")
    for epoch = 1:epochs
        progress!(adam(f, dtrn; gclip=5.0))
        bleu = BLEU(model, dval)
        println("($epoch, $(bleu[1]), $(bleu[2]), $(bleu[3]), $(bleu[4]))")
        if bleu[4] >= bestscore
            Knet.@save savefile1 model vocab bestscore
            bestscore = bleu[4]
        end
        Knet.@save savefile2 model vocab bestscore
    end
end
