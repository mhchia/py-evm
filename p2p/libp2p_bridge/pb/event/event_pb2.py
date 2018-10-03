# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: github.com/ethresearch/sharding-p2p-poc/pb/event/event.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='github.com/ethresearch/sharding-p2p-poc/pb/event/event.proto',
  package='proto.event',
  syntax='proto3',
  serialized_pb=_b('\n<github.com/ethresearch/sharding-p2p-poc/pb/event/event.proto\x12\x0bproto.event\"m\n\x08Response\x12,\n\x06status\x18\x01 \x01(\x0e\x32\x1c.proto.event.Response.Status\x12\x0f\n\x07message\x18\x02 \x01(\t\"\"\n\x06Status\x12\x0b\n\x07SUCCESS\x10\x00\x12\x0b\n\x07\x46\x41ILURE\x10\x01\"?\n\x0eReceiveRequest\x12\x0e\n\x06peerID\x18\x01 \x01(\t\x12\x0f\n\x07msgType\x18\x02 \x01(\x03\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\x0c\"H\n\x0fReceiveResponse\x12\'\n\x08response\x18\x01 \x01(\x0b\x32\x15.proto.event.Response\x12\x0c\n\x04\x64\x61ta\x18\x02 \x01(\x0c\x32O\n\x05\x45vent\x12\x46\n\x07Receive\x12\x1b.proto.event.ReceiveRequest\x1a\x1c.proto.event.ReceiveResponse\"\x00\x62\x06proto3')
)



_RESPONSE_STATUS = _descriptor.EnumDescriptor(
  name='Status',
  full_name='proto.event.Response.Status',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='SUCCESS', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FAILURE', index=1, number=1,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=152,
  serialized_end=186,
)
_sym_db.RegisterEnumDescriptor(_RESPONSE_STATUS)


_RESPONSE = _descriptor.Descriptor(
  name='Response',
  full_name='proto.event.Response',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='status', full_name='proto.event.Response.status', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='message', full_name='proto.event.Response.message', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _RESPONSE_STATUS,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=77,
  serialized_end=186,
)


_RECEIVEREQUEST = _descriptor.Descriptor(
  name='ReceiveRequest',
  full_name='proto.event.ReceiveRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='peerID', full_name='proto.event.ReceiveRequest.peerID', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='msgType', full_name='proto.event.ReceiveRequest.msgType', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data', full_name='proto.event.ReceiveRequest.data', index=2,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=188,
  serialized_end=251,
)


_RECEIVERESPONSE = _descriptor.Descriptor(
  name='ReceiveResponse',
  full_name='proto.event.ReceiveResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='response', full_name='proto.event.ReceiveResponse.response', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data', full_name='proto.event.ReceiveResponse.data', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=253,
  serialized_end=325,
)

_RESPONSE.fields_by_name['status'].enum_type = _RESPONSE_STATUS
_RESPONSE_STATUS.containing_type = _RESPONSE
_RECEIVERESPONSE.fields_by_name['response'].message_type = _RESPONSE
DESCRIPTOR.message_types_by_name['Response'] = _RESPONSE
DESCRIPTOR.message_types_by_name['ReceiveRequest'] = _RECEIVEREQUEST
DESCRIPTOR.message_types_by_name['ReceiveResponse'] = _RECEIVERESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Response = _reflection.GeneratedProtocolMessageType('Response', (_message.Message,), dict(
  DESCRIPTOR = _RESPONSE,
  __module__ = 'p2p.libp2p_bridge.pb.event.event_pb2'
  # @@protoc_insertion_point(class_scope:proto.event.Response)
  ))
_sym_db.RegisterMessage(Response)

ReceiveRequest = _reflection.GeneratedProtocolMessageType('ReceiveRequest', (_message.Message,), dict(
  DESCRIPTOR = _RECEIVEREQUEST,
  __module__ = 'p2p.libp2p_bridge.pb.event.event_pb2'
  # @@protoc_insertion_point(class_scope:proto.event.ReceiveRequest)
  ))
_sym_db.RegisterMessage(ReceiveRequest)

ReceiveResponse = _reflection.GeneratedProtocolMessageType('ReceiveResponse', (_message.Message,), dict(
  DESCRIPTOR = _RECEIVERESPONSE,
  __module__ = 'p2p.libp2p_bridge.pb.event.event_pb2'
  # @@protoc_insertion_point(class_scope:proto.event.ReceiveResponse)
  ))
_sym_db.RegisterMessage(ReceiveResponse)



_EVENT = _descriptor.ServiceDescriptor(
  name='Event',
  full_name='proto.event.Event',
  file=DESCRIPTOR,
  index=0,
  options=None,
  serialized_start=327,
  serialized_end=406,
  methods=[
  _descriptor.MethodDescriptor(
    name='Receive',
    full_name='proto.event.Event.Receive',
    index=0,
    containing_service=None,
    input_type=_RECEIVEREQUEST,
    output_type=_RECEIVERESPONSE,
    options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_EVENT)

DESCRIPTOR.services_by_name['Event'] = _EVENT

# @@protoc_insertion_point(module_scope)