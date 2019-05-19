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


mutable struct ShowAttendAndTell
    convnet
    project
    embedding
    decoder
    predict
    attention
    pdrop
    freeze_convnet
end


function ShowAndTell(; features="conv5", freeze_convnet=true, hiddensize=512,
                     atype=Sloth._atype, vggfile=nothing, pdrop=0.5f0,
                     vocabsize=2541, embedsize=512)
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


function ShowAttendAndTell(; features="conv5", freeze_convnet=true, hiddensize=512,
                           atype=Sloth._atype, vggfile=nothing, pdrop=0.5f0,
                           vocabsize=2541, embedsize=512, attentionsize=512)
    convnet = load_vgg(vggfile; atype=atype, features=features)
    features_dim = 4096
    if features == "conv5" || features == "pool5"
        features_dim = 512
    end
    project = Linear(features_dim, embedsize; atype=atype)
    embedding = Embedding(vocabsize, embedsize; atype=atype)
    decoder = RNN(embedsize, hiddensize)
    predict = Linear(hiddensize, vocabsize; atype=atype)
    attention = Attention(hiddensize, visualsize, visualsize; atype=atype)
    ShowAttendAndTell(convnet, project, embedding, decoder, predict,
                      attention, pdrop, freeze_convnet)
end



function (model::ShowAndTell)(image, words; h=0, c=0,
                              freeze_convnet=model.freeze_convnet)
    model.decoder.h = h; model.decoder.c = c
    visual = extract_features(model, image; freeze_convnet=freeze_convnet)
    visual = model.project(visual)
    visual = dropout(visual, model.pdrop; drop=model.pdrop==0.0)
    visual = reshape(visual, size(visual)..., 1)
    model.decoder(visual)
    x, y = words[:, 1:end-1], words[:, 2:end]
    embed = model.embedding(x)
    embed = dropout(embed, model.pdrop; drop=model.pdrop==0.0)
    hidden = model.decoder(embed)
    hidden = reshape(hidden, size(hidden,1), :)
    hidden = dropout(hidden, model.pdrop; drop=model.pdrop==0.0)
    scores = model.predict(hidden)
    nll(scores, y; average=true)
end


(model::ShowAndTell)(d::TrainLoader) = mean(model(x,y) for (x,y) in d)


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


function extract_features(model::ShowAndTell, x; freeze_convnet=true)
    ndims(x) == 2 && return x
    num_channels = size(x,3)
    y = x
    if num_channels == 3
        y = model.convnet(y)
        y = freeze_convnet ? value(y) : y
    end
    ndims(y) == 2 && return y
    windowsize = size(y,1)
    mat(pool(y; window=windowsize,mode=2))
end


mutable struct Attention
    features_layer
    condition_layer
    scores_layer
end


function Attention(features_dim, condition_dim, output_dim; atype=Sloth._atype)
    features_layer = Linear(features_dim, output_dim; atype=atype)
    condition_layer = Linear(condition_dim, output_dim; atype=atype)
    scores_layer = Linear(output_dim, 1; atype=atype)
    return Attention(features_layer, condition_layer, scores_layer)
end


function (l::Attention)(a, c)
    # features projection - a little bit tricky
    L1, L2, D, B = size(a); L = L1*L2
    a = reshape(a, :, D, B)
    a = permutedims(a, (2,1,3))
    a = reshape(a, D, :)
    a1 = l.features_layer(a)
    a2 = reshape(a1, :, L, B)

    # condition projection
    c1 = l.condition_layer(c)
    c2 = reshape(c1, :, 1, B)

    et = tanh.(a2 .+ c2)
    et = reshape(et, :, B*L)

    scores = reshape(l.scores_layer(et), L, B)
    probs = softmax(scores, dims=2)

    α = reshape(probs, 1, L, B)
    context = α .* a
    context = reshape(context, D, B)
    return α, context
end



function load_vgg(filepath; atype=Sloth._atype, features="conv5")
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
    return convnet
end
