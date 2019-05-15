import Base: iterate, length, rand

mutable struct Dataset
    name
    root_path
    images_path
    features_path
    trn
    val
    tst
end


function Dataset(root_path, features_suffix="default")
    jsonfile = joinpath(abspath(root_path), "flickr8k", "dataset.json")
    jsondata = JSON.parsefile(jsonfile)
    get_split(x) = filter(entry->entry["split"]==x, jsondata["images"])
    trn, val, tst = [get_split(x) for x in ("train","val","test")]
    trn, val = [get_pairs(x) for x in (trn, val)]
    sort!(trn, by=x->length(x[2]), rev=true)
    sort!(val, by=x->length(x[2]), rev=true)
    name = jsondata["dataset"]
    imgpath = abspath(joinpath(root_path, "Flicker8k_Dataset"))
    filename = "$name-features-$features_suffix.jld2"
    featpath = abspath(joinpath(root_path, filename))
    Dataset(name, root_path, imgpath, featpath, trn, val, tst)
end


function build_vocab(d::Dataset)
    words = vcat([sent for (img,sent) in d.trn]...)
    vocab = Vocabulary(words)
end


function get_pairs(images)
    pairs = []
    for image in images
        for sent in image["sentences"]
            push!(pairs, (image["filename"], sent["tokens"]))
        end
    end
    return pairs
end


mutable struct DataLoader
    vocab
    pairs
    images_path
    features_path
    use_features
    batchsize
    atype
    shuffle
    num_instances
    num_batches
end


function DataLoader(vocab, pairs, images_path, features_path;
                    batchsize=128, shuffle=true, use_features=true,
                    atype=Sloth._atype)
    num_instances = length(pairs)
    d, r = divrem(num_instances, batchsize)
    num_batches = r == 0 ? d : d+1
    DataLoader(vocab, pairs, images_path, features_path, use_features,
               batchsize, atype, shuffle, num_instances, num_batches)
end


function iterate(d::DataLoader, state=ifelse(d.shuffle,
                                             randperm(d.num_batches),
                                             [1:d.num_batches...]))
    length(state) == 0 && return nothing
    this = popfirst!(state)
    from = (this-1)*d.batchsize+1
    to = min(from+d.batchsize-1, d.num_instances)
    pairs = d.pairs[from:to]
    filenames = map(first, pairs)
    tokens = map(last, pairs)
    longest = length(first(tokens))
    batchsize = length(tokens)
    words = zeros(Int, batchsize, longest+1)*length(d.vocab)
    # return (tokens, state)
    # @show size(words), batchsize, longest
    for i = 1:batchsize, t = 1:longest
        t >= length(tokens[i]) && continue
        words[i,t+1] = word2index(d.vocab, tokens[i][t])
    end
    sos = d.vocab.w2i["<sos>"]
    words[:,1] .= sos

    images = []
    if false # d.use_features

    else
        for filename in filenames
            filepath = joinpath(d.images_path, filename)
            push!(images, Sloth.imgdata(filepath))
        end
        images = cat(images..., dims=4)
    end

    return ((filenames, images, words), state)
end
