import Sloth: matconvnet, VGG, load_weights!


mutable struct ShowAndTell
    convnet
    project
    embedding
    decoder
    predict
    pdrop
    freeze_convnet
end


function ShowAndTell(; features="conv5", freeze_convnet=true, hiddensize=512,
                     atype=Sloth._atype, vggfile=nothing, pdrop=0.5f0,
                     vocabsize=100, embedsize=512)
    convnet = load_vgg(vggfile; atype=atype, features=features)
    features_dim = 4096
    if features == "conv5" || features == "pool5"
        features_dim = 512
    end
    project = Linear(features_dim, embedsize; atype=atype)
    embedding = Embedding(vocabsize, embedsize; atype=atype)
    decoder = RNN(embedsize, hiddensize)
    predict = Linear(hiddensize, vocabsize; atype=atype)
    ShowAndTell(convnet, project, embedding, decoder, predict,
                pdrop, freeze_convnet)
end


function (model::ShowAndTell)(image, words; h=0, c=0)
    model.decoder.h = h; model.decoder.c = c
    visual = extract_features(model, image)
    visual = model.project(visual)
    visual = reshape(visual, size(visual)..., 1)
    model.decoder(visual)
    x, y = words[:, 1:end-1], words[:, 2:end]
    embed = model.embedding(x)
    hidden = model.decoder(embed)
    hidden = reshape(hidden, size(hidden,1), :)
    scores = model.predict(hidden)
    return nll(scores, y)
end


function (model::ShowAndTell)(image; maxlen=20, h=0, c=0,
                              start_token=1, end_token=10)
    model.decoder.h = h; model.decoder.c = c
    visual = extract_features(model, image)
    visual = model.project(visual)
    visual = reshape(visual, size(visual)..., 1)
    hidden = model.decoder(visual)
    words = Any[start_token]

    for t = 1:maxlen
        embed = model.embedding(words[end])
        hidden = model.decoder(embed)
        hidden = reshape(hidden, size(hidden,1), :)
        scores = model.predict(hidden)
        push!(words, argmax(Array(scores))[1])
    end
    return words
end


function extract_features(model::ShowAndTell, x)
    ndims(x) == 2 && return x
    y = model.convnet(x)
    y = model.freeze_convnet ? value(y) : y
    ndims(y) == 2 && return y
    windowsize = size(y,1)
    mat(pool(y; window=windowsize,mode=2))
end



function load_vgg(filepath; atype=Sloth._atype, features="conv5")
    convnet = VGG(; atype=atype)
    vggmat = matconvnet(filepath)
    load_weights!(convnet, vggmat)
    drop_last_nitems = 2
    if features == "pool5"
        drop_last_nitems = 5
    elseif features == "conv5"
        drop_last_nitems = 6
    end
    for i = 1:drop_last_nitems; pop!(convnet.layers); end
    return convnet
end
