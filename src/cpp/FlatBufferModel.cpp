class FlatBufferModel {

    //Builds a model from a file. Returns null if the file cannot be read.
    static std::unique_ptr<FlatBufferModel> BuildFromFile(
        const char* filename,
        ErrorReporter* error_reporter);

    //Builds a model based on  a pre-loaded flatbuffer. Caller retains ownership of the buffer.

    static std::unique_ptr<FlatBufferModel> BuildFromBuffer(
        const char* buffer,
        size_t buffer_size,
        ErrorReporter* error_reporter);
};