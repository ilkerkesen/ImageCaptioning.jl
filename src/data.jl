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
    trn, val, tst = [get_instances(x) for x in (trn, val, tst)]
    # sort!(val, by=x->length(x[2]), rev=true)
    name = jsondata["dataset"]
    imgpath = abspath(joinpath(root_path, "Flicker8k_Dataset"))
    filename = "$name-features-$features_suffix.jld2"
    featpath = abspath(joinpath(root_path, filename))
    Dataset(name, root_path, imgpath, featpath, trn, val, tst)
end


function extract_features(d::Dataset, model)
    isfile(d.features_path) && return
    filenames = mapreduce(s->map(first, s), vcat, (d.trn, d.val, d.tst))
    jldopen(d.features_path, "w") do f
        @showprogress for (i, filename) in enumerate(filenames)
            filepath = joinpath(d.images_path, filename)
            img = oftype(model.convnet.layers[1].w, Sloth.imgdata(filepath))
            features = model.convnet(img)
            f[filename] = Array(features)
        end
    end
end


function build_vocab(d::Dataset)
    words = vcat([vcat(x...) for (_,x) in d.trn]...)
    vocab = Vocabulary(words)
end


function get_pairs(instances)
    pairs = []
    for (filename, sentences) in instances
        for sent in sentences
            push!(pairs, (filename, sent))
        end
    end
    return pairs
end


function get_instances(images)
    instances = []
    for image in images
        tokens = [x["tokens"] for x in image["sentences"]]
        push!(instances, (image["filename"], tokens))
    end
    return instances
end


mutable struct TrainLoader
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


function TrainLoader(vocab, instances, images_path, features_path;
                    batchsize=128, shuffle=true, use_features=true,
                    atype=Sloth._atype, sorting=true)
    pairs = get_pairs(instances)
    sorting && sort!(pairs, by=x->length(x[2]), rev=true)
    num_instances = length(pairs)
    d, r = divrem(num_instances, batchsize)
    num_batches = r == 0 ? d : d+1
    TrainLoader(vocab, pairs, images_path, features_path, use_features,
               batchsize, atype, shuffle, num_instances, num_batches)
end


function iterate(d::TrainLoader, state=ifelse(d.shuffle,
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
    words = ones(Int, batchsize, longest+2)*length(d.vocab)
    for i = 1:batchsize, t = 1:longest
        t > length(tokens[i]) && continue
        words[i,t+1] = word2index(d.vocab, tokens[i][t])
    end
    sos = d.vocab.w2i["<sos>"]
    words[:,1] .= sos

    images = []
    if d.use_features # d.use_features
        jldopen(d.features_path, "r") do f
            for file in filenames; push!(images, f[file]); end
        end
    else
        for filename in filenames
            filepath = joinpath(d.images_path, filename)
            push!(images, Sloth.imgdata(filepath))
        end
    end
    images = d.atype(cat(images..., dims=4))

    return ((images, words, filenames), state)
end


length(d::TrainLoader) = d.num_batches


mutable struct EvalLoader
    vocab
    instances
    images_path
    features_path
    use_features
    atype
    num_instances
end


function EvalLoader(vocab, instances, images_path, features_path;
                    use_features=true, atype=Sloth._atype)
    num_instances = length(instances)
    EvalLoader(vocab, instances, images_path, features_path, use_features, atype,
               num_instances)
end


function iterate(d::EvalLoader, state=1:d.num_instances)
    length(state) == 0 && return nothing
    filename, references = d.instances[first(state)]
    state = state[2:end]
    image = nothing
    if d.use_features
        jldopen(d.features_path, "r") do f
            image = f[filename]
        end
    else
        filepath = joinpath(d.images_path, filename)
        image = Sloth.imgdata(filepath)
    end
    image = d.atype(image)
    return ((image, references, filename), state)
end


length(d::EvalLoader) = d.num_instances
