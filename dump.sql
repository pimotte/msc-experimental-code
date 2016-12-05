.mode csv
.headers on
.separator ;
.out /home/pim/school/msc/algorithms/multichain_exp.csv
SELECT hex(public_key_requester), hex(public_key_responder), up, down,total_up_requester, total_down_requester, sequence_number_requester, hex(previous_hash_requester),hex(signature_requester), hex(hash_requester),total_up_responder, total_down_responder, sequence_number_responder, hex(previous_hash_responder), hex(signature_responder), hex(hash_responder), insert_time FROM multi_chain;
