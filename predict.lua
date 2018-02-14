require 'paths'
require 'csvigo'
paths.dofile('util.lua')
paths.dofile('img.lua')

if #arg == 0 then
  print("No input images given")
  return
end

images = arg
preds = torch.Tensor(#images, 16, 2)

m = torch.load('umich-stacked-hourglass.t7')   -- Load pre-trained model

xlua.progress(0, #images)

points = {}

for i = 1, #images do
  -- Load image
  local img = image.load(images[i])

  -- Normalize and center
  local scale = 4.5
  local center = {568, 596}
  local input = crop(img, center, scale, 0, 256)

  -- Predict pose
  local out = m:forward(input:view(1,3,256,256):cuda())
  cutorch.synchronize()
  local hm = out[#out][1]:float()
  hm[hm:lt(0)] = 0
  local preds_hm, preds_img = getPreds(hm, center, scale)

  -- Store result
  points[#points + 1] = torch.totable(preds_img:view(preds_img:nElement()))

  -- Continue
  xlua.progress(i, #images)
  collectgarbage()
end

csvigo.save('output.csv', points)
