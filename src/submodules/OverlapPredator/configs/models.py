architectures = dict()
architectures['indoor'] = [
    'simple',
    'resnetb',
    'resnetb_strided',
    
    'resnetb',
    'resnetb',
    'resnetb_strided',
    
    'resnetb',
    'resnetb',
    'resnetb_strided',
    
    'resnetb',
    'resnetb',
    
    'nearest_upsample',
    'unary',
    'nearest_upsample',
    'unary',
    'nearest_upsample',
    'last_unary'
]

architectures['kitti'] = [
    'simple',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'nearest_upsample',
    'unary',
    'nearest_upsample',
    'unary',
    'nearest_upsample',
    'last_unary'
]

architectures['modelnet'] = [
    'simple',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'nearest_upsample',
    'unary',
    'unary',
    'nearest_upsample',
    'unary',
    'last_unary'
]


"""
                    PREDATOR
==================================================================
ENCODER             ----------------------------------[[4_998_080]]

BLOCK 1
'simple',           ----------------------------------------[[960]]
    - KPConv [960]
    - leaky_relu [0]
'resnetb',          -------------------------------------[[29_696]]
        main                |       shortcut
    - unary1 [2_048]        |   - unary [8_192]
    - KPConv [15_360]       |
    - batch_norm [0]        |
    - leaky_relu [0]        |
    - unary2 [4_096]        |
'resnetb_strided',  -------------------------------------[[23_552]]
        main                |       shortcut
    - unary1 [4_096]        |   - max_pool [0]
    - KPConv [15_360]       |   - unary_shortcut [0]
    - leaky_relu [0]        |
    - unary2 [4_096]        |
==================================================================
BLOCK 2 
'resnetb',          -----------------------------------[[118_784]]
        main                |       shortcut
    - unary1 [8_192]        |   - unary [32_768]
    - KPConv [61_440]       |
    - batch_norm [0]        |
    - leaky_relu [0]        |
    - unary2 [16_384]       |
'resnetb',          ------------------------------------[[94_208]]
        main                |       shortcut
    - unary1 [16_384]       |   - unary [0] ?????
    - KPConv [61_440]       |
    - batch_norm [0]        |
    - leaky_relu [0]        |
    - unary2 [16_384]       |
'resnetb_strided',  ------------------------------------[[94_208]]
        main                |       shortcut
    - unary1 [16_384]       |   - max_pool [0]
    - KPConv [61_440]       |   - unary_shortcut [0]
    - leaky_relu [0]        |
    - unary2 [16_384]       |
==================================================================
BLOCK 3
'resnetb',          ------------------------------------[[475_136]]
        main                |       shortcut
    - unary1 [32_768]       |   - unary [131_072]
    - KPConv [245_760]      |
    - batch_norm [0]        |
    - leaky_relu [0]        |
    - unary2 [65_536]       |
'resnetb',          ------------------------------------[[376_832]]
        main                |       shortcut
    - unary1 [65_536]       |   - unary [0]
    - KPConv [24_5760]      |
    - batch_norm [0]        |
    - leaky_relu [0]        |
    - unary2 [65_536]       |
'resnetb_strided',  ------------------------------------[[376_832]]
        main                |       shortcut
    - unary1 [65_536]       |   - max_pool [0]
    - KPConv [245_760]      |   - unary_shortcut [0]
    - leaky_relu [0]        |
    - unary2 [65_536]       |
==================================================================
BLOCK 4
'resnetb',          ----------------------------------[[1_900_544]]
        main                |       shortcut
    - unary1 [131_072]      |   - unary [524_288]
    - KPConv [983_040]      |
    - batch_norm [0]        |
    - leaky_relu [0]        |
    - unary2 [262_144]      |
'resnetb',          ----------------------------------[[1_507_328]]
        main                |       shortcut
    - unary1 [262_144]      |   - unary [0]
    - KPConv [983_040]      |
    - batch_norm [0]        |
    - leaky_relu [0]        |
    - unary2 [262_144]      |
==================================================================
BOTTLENECK             -------------------------------[[2_296_321]]

nn.conv1d # bottle   -----------------------------------[[262_400]]
GCN                  ---------------------------------[[1_967_872]]
nn.conv1d # proj_gnn ------------------------------------[[65_792]]
nn.conv1d # proj_score -------------------------------------[[257]]

==================================================================
DECODER             ------------------------------------[[130_498]]

'nearest_upsample', ------------------------------------------[[0]]
'unary',            -------------------------------------[[99_330]]
'nearest_upsample', ------------------------------------------[[0]]
'unary',            -------------------------------------[[24_640]]
'nearest_upsample', ------------------------------------------[[0]]
'last_unary'        --------------------------------------[[6_528]]
"""