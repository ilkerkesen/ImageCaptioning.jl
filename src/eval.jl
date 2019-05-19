function BLEU(model, data; o...)
    path = abspath(joinpath(DIR, "..", "eval"))
    i2w = data.vocab.i2w
    hyps = open(joinpath(path, "hypothesis"), "w")
    fns = open(joinpath(path, "filenames"), "w")
    refs = [open(joinpath(path,"reference$i"), "w") for i=0:4]
    @showprogress for (image, references, filename) in data
        hyp = [map(i->i2w[i], model(image))..., ".\n"]
        hyp = filter(w->!(w in ("<unk>","<pad>","<eos>","<sos>")), hyp)
        hyp = join(hyp, " ")
        write(hyps, hyp)
        write(fns, "$(filename)\n")
        for (i,tokens) in enumerate(references)
            ref = join([tokens..., ".\n"], " ")
            write(refs[i], ref)
        end
    end
    close(hyps)
    close(fns)
    map(close, refs)
    out = read(`$path/eval.sh $path`, String)
    out = out[8:findfirst(isequal('('),out)-2]
    return map(x->parse(Float64, x), split(out,"/"))
end
