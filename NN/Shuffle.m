 function mat_shuffled = Shuffle(mat)
     time = clock;
     seed = time(6);
     rng(seed);
     indexes = randi(4,4,1)';
     mat_shuffled = zeros(4,4);
     for i = 1:4
         new_row = mat(indexes(i),:);
         mat_shuffled(i,:)= new_row;
     end
 end

