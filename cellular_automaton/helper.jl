function rle2txt(rle::String,cell_size::Tuple{T,T}) where T <: Int
    # convert run length encoding of Conway Game of Life to a matrix of 0s and 1s
    # use '0' for 'o' and '1' for 'b'
    plain = zeros(Int,cell_size)
    rol = 1
    col = 1
    for i in 1:length(rle)-1
        cur_num = 0
        if rle[i] in ['0','1','2','3','4','5','6','7','8','9']
            cur_num = cur_num*10 + parse(Int,rle[i])
        else
            cur_num = cur_num == 0 ? 1 : cur_num
            if rle[i] == 'o'
                plain[rol,col:col+cur_num] .= 1
                col += cur_num + 1
                cur_num = 0
            elseif rle[i] == 'b'
                col += cur_num + 1
                cur_num = 0
            elseif rle[i] == '$'
                rol += cur_num
                col = 1
                cur_num = 0
            end
        end

    end
    return plain
end
