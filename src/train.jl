function train(;
               arch=ShowAndTell,
               name="exp1",
               hiddensize=512,
               embedsize=512,
               batchsize=64,
               seed=-1,
               datadir=joinpath(DIR, "../data"),
               epochs=50,
               vggfile=joinpath(datadir, "imagenet-vgg-verydeep-16.mat"),
               atype=Sloth._atype,
               loadfile=nothing,
               pdrop=0.5f0)
    seed != -1 && Knet.seed!(seed)
    data = Dataset(datadir)
    vocab = net = start = history = bestscore = nothing
    if loadfile == nothing
        vocab = build_vocab(data)
        net = arch(; hiddensize=hiddensize, embedsize=embedsize, atype=atype,
                   vggfile=vggfile, vocabsize=length(vocab.w2i))
        start = 1
        history = []
    else
        Knet.@load loadfile net vocab history epoch
        start = epoch+1
        bestscore = maximum(map(hi->hi[end], history))
    end
    dtrn = TrainLoader(
        vocab, data.trn, data.images_path, data.features_path;
        batchsize=batchsize, atype=atype)
    dval = EvalLoader(vocab, data.val, data.images_path, data.features_path;
                      atype=atype)

    if length(history) == 0
        bleu = BLEU(net, dval, "val")
        push!(history, bleu)
        bestscore = bleu[end]
    end
    f = (img,txt,_)->loss(net,img,txt; pdrop=pdrop)
    modelpath = abspath(joinpath(DIR, "../models"))
    bestfile = joinpath(modelpath, "$(name)-best.jld2")
    checkpoint = joinpath(modelpath, "$(name)-last.jld2")
    for epoch = start:epochs
        progress!(adam(f, dtrn; gclip=5.0))
        bleu = BLEU(net, dval, "val")
        push!(history, bleu)
        println("($epoch, $(bleu[1]), $(bleu[2]), $(bleu[3]), $(bleu[4]))")
        if bleu[4] >= bestscore
            Knet.@save bestfile net vocab history epoch
            bestscore = bleu[4]
        end
        Knet.@save checkpoint net vocab history epoch
    end
end
