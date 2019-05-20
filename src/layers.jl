mutable struct Attention
    features_layer
    condition_layer
    scores_layer
end


function Attention(features_dim::Int, condition_dim::Int, output_dim::Int;
                   atype=Sloth._atype)
    features_layer = Linear(features_dim, output_dim; atype=atype)
    condition_layer = Linear(condition_dim, output_dim; atype=atype)
    scores_layer = Linear(output_dim, 1; atype=atype)
    return Attention(features_layer, condition_layer, scores_layer)
end


function (l::Attention)(a, c)
    # features projection - a little bit tricky
    D, B = size(a)[end-1:end] # size(a) = (L1, L2, D, B) or (14,14,512,32)
    a = reshape(a, :, D, B)   # size(a) = (L, D, B) or (196, 512, 32)
    L = size(a,1) # L=196
    a0 = permutedims(a, (2,1,3)) # size(a0) = (D, L, B) or (512, 196, 32)
    a0 = reshape(a0, D, :) # size(a0) = (D, L*B) or (512, 196*32)
    a1 = l.features_layer(a0) # size(a1) = (D, L*B)
    a2 = reshape(a1, :, L, B) # size(a2) = (D, L, B)

    # condition projection
    c0 = reshape(c, :, B)
    c1 = l.condition_layer(c0) # size(c1) = (D, B)
    c2 = reshape(c1, :, 1, B) # size(c2) = (D, 1, B)

    et = tanh.(a2 .+ c2) # size(et) = (D, L, B)
    et = reshape(et, :, B*L) # size(et) =

    scores = reshape(l.scores_layer(et), L, B)
    probs = softmax(scores, dims=1)

    α = reshape(probs, L, 1, B)
    context = α .* a
    context = sum(context, dims=1)

    α = reshape(α, L, B)
    context = reshape(context, D, B)
    return α, context
end
