import Sloth: matconvnet, VGG, load_weights!


abstract type CaptionNetwork end


mutable struct ShowAndTell <: CaptionNetwork
    convnet
    project
    embedding
    decoder
    predict
    pdrop
    freeze_convnet
end


mutable struct ShowAttendAndTell <: CaptionNetwork
    convnet
    embedding
    decoder
    predict
    attention
    pdrop
    freeze_convnet
end


function ShowAndTell(; features="conv5", freeze_convnet=true, hiddensize=512,
                     atype=Sloth._atype, vggfile=nothing, pdrop=0.0f0,
                     vocabsize=2541, embedsize=512)
    convnet = load_vgg(
        vggfile; atype=atype, features=features, freeze=freeze_convnet)
    features_dim = 4096
    if features == "conv5" || features == "pool5"
        features_dim = 512
    end
    project = Linear(features_dim, embedsize; atype=atype)
    embedding = Embedding(vocabsize, embedsize; atype=atype)
    usegpu = atype <: KnetArray
    dataType = eltype(atype)
    decoder = RNN(embedsize, hiddensize; usegpu=usegpu, dataType=dataType)
    predict = Linear(hiddensize, vocabsize; atype=atype)
    ShowAndTell(convnet, project, embedding, decoder, predict,
                pdrop, freeze_convnet)
end


function ShowAttendAndTell(; features="conv5", freeze_convnet=true, hiddensize=512,
                           atype=Sloth._atype, vggfile=nothing, pdrop=0.5f0,
                           vocabsize=2541, embedsize=512, attentionsize=512)
    convnet = load_vgg(
        vggfile; atype=atype, features=features, freeze=freeze_convnet)
    features_dim = 512
    embedding = Embedding(vocabsize, embedsize; atype=atype)
    usegpu = atype <: KnetArray
    dataType = eltype(atype)
    decoder = RNN(embedsize+features_dim, hiddensize; usegpu=usegpu, dataType=dataType)
    predict = Linear(hiddensize, vocabsize; atype=atype)
    attention = Attention(features_dim, hiddensize, features_dim; atype=atype)
    ShowAttendAndTell(convnet, embedding, decoder, predict, attention,
                      pdrop, freeze_convnet)
end


function loss(net::CaptionNetwork, image, words; pdrop=net.pdrop)
    visual = extract_features(net, image; pdrop=pdrop)
    initstate!(net, visual)
    input_words, output_words = words[:, 1:end-1], words[:, 2:end]
    scores, _ = decode(net, visual, input_words; pdrop=pdrop)
    nll(scores, output_words)
end




function extract_features(net::ShowAndTell, image; pdrop=net.pdrop)
    x = image
    ndims(x) == 2 && return x
    if ndims(x) == 4 && size(x,3) == 3
        x = net.convnet(x)
    end
    if ndims(x) != 2
        D, B = size(x)[end-1:end]
        wsize = size(x,1)
        x = pool(x; window=wsize, mode=2)
        x = reshape(x, D, B)
    end
    x = dropout(x, pdrop; drop=pdrop>0.0)
    x = net.project(x)
end


function extract_features(net::ShowAttendAndTell, image; pdrop=net.pdrop)
    size(image,3) != 3 && return image
    features = net.convnet(image)
end


function initstate!(net::ShowAndTell, visual)
    net.decoder.h = 0
    net.decoder.c = 0
    net.decoder(visual)
end


function initstate!(net::ShowAttendAndTell, visual)
    D, B = size(visual)[end-1:end]
    hidden = mean(reshape(visual, :, D, B), dims=1)
    hidden = reshape(hidden, D, B, 1)
    net.decoder.h = hidden
    net.decoder.c = hidden
end


function decode(net::ShowAndTell, visual, input_words; pdrop=net.pdrop)
    # net.decoder(visual)
    embed = net.embedding(input_words)
    embed = dropout(embed, pdrop; drop=pdrop>0.0)
    hidden = net.decoder(embed)
    hidden = reshape(hidden, size(hidden,1), :)
    hidden = dropout(hidden, pdrop; drop=pdrop>0.0)
    scores = net.predict(hidden)
    return scores, nothing
end


function decode(net::ShowAttendAndTell, visual, input_words; pdrop=net.pdrop)
    T = size(input_words, 2)
    B = size(visual)[end]
    hiddens = []
    αs = []
    for t = 1:T
        h, α = step!(net, visual, input_words[:,t]; pdrop=0.0)
        push!(hiddens, reshape(h, :, B))
        push!(αs, α)
    end
    hidden = cat(hiddens..., dims=2)
    hidden = dropout(hidden, pdrop; drop=pdrop>0.0)
    scores = net.predict(hidden)
    return scores, αs
end


function decode(net::CaptionNetwork, vocab::Vocabulary, visual, maxlen=20)
    obsolete = Any["<sos>","<eos>","<pad>","<unk>"]
    obsolete = Any[vocab.w2i[w] for w in obsolete]
    words = Any[first(obsolete)]
    other = []

    for t = 1:maxlen
        prev_word = words[end]
        hidden, o = step!(net, visual, prev_word; pdrop=0.0)
        scores = net.predict(hidden)
        next_word = argmax(Array(scores))[1]
        push!(words, next_word)
        push!(other, o)
    end

    words = filter(i->!(i in obsolete), words)
    words = [map(w->vocab.i2w[w], words)..., "."]
    sentence = join(words, " ")
    return sentence, other
end


function step!(net::ShowAndTell, visual, word; pdrop=net.pdrop)
    embed = net.embedding(word)
    embed = dropout(embed, pdrop; drop=pdrop>0.0)
    net.decoder(embed)
    hidden = net.decoder.h
    return hidden, nothing
end


function step!(net::ShowAttendAndTell, visual, word; pdrop=net.pdrop)
    embed = net.embedding(word)
    embed = dropout(embed, pdrop; drop=pdrop>0.0)
    α, ctx = net.attention(visual, net.decoder.h)
    input = vcat(embed, ctx)
    net.decoder(input)
    hidden = net.decoder.h
    return hidden, α
end


function generate(net::CaptionNetwork, vocab::Vocabulary, image; maxlen=20)
    visual = extract_features(net, image)
    initstate!(net, visual)
    decode(net, vocab, visual, maxlen)
end


# (net::ShowAndTell)(d::TrainLoader) = mean(model(x,y) for (x,y) in d)


function load_vgg(filepath; atype=Sloth._atype, features="conv5", freeze=true)
    convnet = VGG(; atype=atype)
    vggmat = matconvnet(filepath)
    Sloth.load_weights!(convnet, vggmat)
    drop_last_nitems = 2
    if features == "pool5"
        drop_last_nitems = 5
    elseif features == "conv5"
        drop_last_nitems = 6
    end
    for i = 1:drop_last_nitems; pop!(convnet.layers); end
    !freeze && return convnet
    for i in length(convnet.layers)
        if isa(convnet.layers[i], Conv) || isa(convnet.layers[i], FullyConnected)
            convnet.layers[i].w = w.value
            convnet.layers[i].b = b.value
        end
    end
    return convnet
end
