function BLEU(net::CaptionNetwork, data, data_split="val"; maxlen=20)
    path = abspath(EVAL_DIR, data_split)
    captions = []
    @showprogress for (img,ref,fn) in data
        caption = generate(net, data.vocab, img)[1]
        push!(captions, caption)
    end
    # captions = (generate(net, data.vocab, img) for (img,ref,fn) in data)
    captions = join(captions, "\n")
    open(joinpath(path, "hypothesis"), "w") do f
        write(f, captions)
    end
    captions = 0
    out = read(`sh $DIR/eval.sh $path`, String)
    out = out[8:findfirst(isequal('('),out)-2]
    return map(x->parse(Float64, x), split(out,"/"))
end
