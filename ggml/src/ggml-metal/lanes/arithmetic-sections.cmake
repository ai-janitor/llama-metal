# The sections of data/arithmetic.tsv and their header rows, for lanes/gen-tables.cmake and act-copy/gen-act-copy.cmake.
set(arithmetic_sections capabilities lanes copies)
set(arithmetic_capabilities_header "card;capability;in_hardware;reached;correct;ratio;source")
set(arithmetic_lanes_header "lane;source;file;family;route;weight;activation;operand;holds;entry;capability;class;origin;default;seam_us;id;reason;test")
set(arithmetic_copies_header "card;family;route;copy;applies;min_rereads;max_bytes;reason;test")
