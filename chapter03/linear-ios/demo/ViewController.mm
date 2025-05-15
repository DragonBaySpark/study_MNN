//
//  ViewController.mm
//  MNN
//
//  Created by MNN on 2019/02/03.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#import "ViewController.h"
#import <Metal/Metal.h>

#import <AVFoundation/AVFoundation.h>
#import <MNN/HalideRuntime.h>
#import <MNN/MNNDefine.h>
#import <MNN/ErrorCode.hpp>
#import <MNN/ImageProcess.hpp>
#import <MNN/Interpreter.hpp>
#import <MNN/Tensor.hpp>
#define MNN_METAL
#import <MNN/MNNSharedContext.h>

#include <thread>
typedef struct {
    float value;
    int index;
} LabeledElement;

static int CompareElements(const LabeledElement *a, const LabeledElement *b) {
    if (a->value > b->value) {
        return -1;
    } else if (a->value < b->value) {
        return 1;
    } else {
        return 0;
    }
}

struct PretreatInfo {
    int outputSize[4];
    float mean[4];
    float normal[4];
    float inputSize[4];
    float matrix[16];
};
/**
 * GPU缓存结构体，用于管理Metal GPU相关的资源
 * 包含纹理缓存、设备、计算管线、常量缓冲区等Metal对象
 */
struct GpuCache {
    CVMetalTextureCacheRef _textureCache;  // Metal纹理缓存引用，用于管理纹理资源
    id<MTLDevice> _device;                 // Metal设备对象，代表物理GPU
    id<MTLComputePipelineState> _pretreat; // 预处理计算管线状态
    id<MTLFunction> _function;             // Metal着色器函数
    id<MTLBuffer> _constant;               // 常量缓冲区，用于存储预处理参数
    id<MTLCommandQueue> _queue;            // 命令队列，用于向GPU提交命令
    
    /**
     * 构造函数：初始化所有Metal相关资源
     */
    GpuCache() {
        // 创建默认的Metal设备（GPU）
        _device = MTLCreateSystemDefaultDevice();
        
        // 创建Metal纹理缓存
        CVReturn res = CVMetalTextureCacheCreate(nil, nil, _device, nil, &_textureCache);
        FUNC_PRINT(res);
        
        // 获取默认的Metal库，包含编译好的着色器函数
        id<MTLLibrary> library = [_device newDefaultLibrary];
        // 获取名为"pretreat"的预处理着色器函数
        _function = [library newFunctionWithName:@"pretreat"];
        
        // 创建计算管线状态
        NSError* error = nil;
        _pretreat = [_device newComputePipelineStateWithFunction:_function error:&error];
        
        // 创建常量缓冲区，大小为PretreatInfo结构体的大小
        _constant = [_device newBufferWithLength:sizeof(PretreatInfo) 
                                       options:MTLCPUCacheModeDefaultCache];
        
        // 创建命令队列
        _queue = [_device newCommandQueue];
    }
    
    /**
     * 析构函数：清理资源
     * TODO: 需要实现资源的释放
     */
    ~GpuCache() {
        // 这里应该添加资源释放代码
        // 释放 _textureCache, _device, _pretreat, _function, _constant, _queue 等
    }
};
@interface Model : NSObject {
    std::shared_ptr<MNN::Interpreter> _net;
    MNN::Session *_session;
    std::mutex _mutex;
    MNNForwardType _type;
    MNN::Tensor* _input;
    MNN::Tensor* _output;
    std::shared_ptr<GpuCache> _cache;
}
@property (strong, nonatomic) UIImage *defaultImage;
@property (strong, nonatomic) NSArray<NSString *> *labels;

@end

@implementation Model
/**
 * 设置MNN推理引擎的运行类型和线程数
 * @param type 运行类型（CPU或Metal GPU）
 * @param threads 线程数量，用于CPU模式下的并行计算
 */
- (void)setType:(MNNForwardType)type threads:(NSUInteger)threads {
    // 加锁保护，防止多线程并发访问导致的问题
    std::unique_lock<std::mutex> _l(_mutex);
    
    NSLog(@"setType: %d, threads: %lu", type, (unsigned long)threads);
    // 如果已存在会话，先释放旧会话
    if (_session) {
        _net->releaseSession(_session);
    }
    
    // 如果GPU缓存未初始化，创建新的GPU资源缓存
    if (nullptr == _cache) {
        _cache.reset(new GpuCache);
    }
    
    // 配置MNN推理引擎的运行参数
    MNN::ScheduleConfig config;
    config.type      = type;      // 设置运行类型（CPU/GPU）
    config.numThread = (int)threads;  // 设置CPU线程数
    
    // 根据运行类型进行不同的会话创建
    if (type == MNN_FORWARD_METAL) {
        // Metal GPU模式下的特殊配置
        MNN::BackendConfig bnConfig;
        MNNMetalSharedContext context;
        // 设置Metal上下文，复用已创建的设备和命令队列
        context.device = _cache->_device;
        context.queue = _cache->_queue;
        bnConfig.sharedContext = &context;
        config.backendConfig = &bnConfig;
        _session = _net->createSession(config);
    } else {
        // CPU模式下的标准配置
        _session = _net->createSession(config);
    }
    
    // 获取会话的输入输出张量
    _input = _net->getSessionInput(_session, nullptr);   // 获取输入张量
    _output = _net->getSessionOutput(_session, nullptr); // 获取输出张量
    _type = type;  // 保存当前运行类型
}

/**
 * 执行模型推理的基准测试
 * @param cycles 测试循环次数，用于计算平均推理时间
 * @return 返回包含平均推理时间的字符串，如果初始化失败返回nil
 */
- (NSString *)benchmark:(NSInteger)cycles {
    // 加锁保护，确保多线程安全
    std::unique_lock<std::mutex> _l(_mutex);
    
    // 检查网络和会话是否正确初始化
    if (!_net || !_session) {
        return nil;
    }
    
    // 获取模型的输出张量并创建副本
    MNN::Tensor *output = _net->getSessionOutput(_session, nullptr);
    MNN::Tensor copy(output);
    
    // 获取模型的输入张量并创建缓存
    auto input = _net->getSessionInput(_session, nullptr);
    MNN::Tensor tensorCache(input);
    
    // 将输入数据复制到主机内存
    input->copyToHostTensor(&tensorCache);

    // 记录开始时间
    NSTimeInterval begin = NSDate.timeIntervalSinceReferenceDate;
    
    // 执行多次推理，用于计算平均时间
    for (int i = 0; i < cycles; i++) {
        // 将缓存的输入数据复制到输入张量
        input->copyFromHostTensor(&tensorCache);
        // 运行一次推理会话
        _net->runSession(_session);
        // 将输出结果复制到输出张量副本
        output->copyToHostTensor(&copy);
    }
    
    // 计算总耗时
    NSTimeInterval cost = NSDate.timeIntervalSinceReferenceDate - begin;
    
    // 格式化输出结果，计算平均每次推理的耗时（毫秒）
    NSString *string = @"";
    return [string stringByAppendingFormat:@"time elapse: %.3f ms", cost * 1000.f / cycles];
}

- (NSString *)inferNoLock:(NSInteger)cycles {
    if (!_net || !_session) {
        return nil;
    }
    // run
    NSTimeInterval begin = NSDate.timeIntervalSinceReferenceDate;
    // you should set input data for each inference
    _net->runSession(_session);

    MNN::Tensor *output = _net->getSessionOutput(_session, nullptr);
    MNN::Tensor copy(output);
    output->copyToHostTensor(&copy);
    NSTimeInterval cost = NSDate.timeIntervalSinceReferenceDate - begin;

    // result
    float *data = copy.host<float>();
    LabeledElement objects[1000];
    for (int i = 0; i < 1000; i++) {
        objects[i].value = data[i];
        objects[i].index = i;
    }
    qsort(objects, 1000, sizeof(objects[0]), (int (*)(const void *, const void *))CompareElements);

    // to string
    NSString *string = @"";
    for (int i = 0; i < 3; i++) {
        string = [string stringByAppendingFormat:@"%@: %f\n", _labels[objects[i].index], objects[i].value];
    }
    return [string stringByAppendingFormat:@"time elapse: %.3f ms", cost * 1000.f / 1.0f];
}

- (NSString *)infer:(NSInteger)cycles {
    std::unique_lock<std::mutex> _l(_mutex);
    return [self inferNoLock:cycles];
}


- (NSString *)inferImage:(UIImage *)image cycles:(NSInteger)cycles {
    return [self infer:cycles];
}
- (NSString *)inferBuffer:(CMSampleBufferRef)buffer {
    return [self infer:1];
}
@end

#pragma mark -
@interface MobileNetV2 : Model
@end

@implementation MobileNetV2
- (instancetype)init {
    self = [super init];
    if (self) {
        NSString *labels  = [[NSBundle mainBundle] pathForResource:@"synset_words" ofType:@"txt"];
        NSString *lines   = [NSString stringWithContentsOfFile:labels encoding:NSUTF8StringEncoding error:nil];
        self.labels       = [lines componentsSeparatedByString:@"\n"];
        self.defaultImage = [UIImage imageNamed:@"testcat.jpg"];

        NSString *model = [[NSBundle mainBundle] pathForResource:@"mobilenet_v2.caffe" ofType:@"mnn"];
        _net            = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model.UTF8String));
    }
    return self;
}
/**
 * 对输入图像进行推理
 * @param image 输入的UIImage图像
 * @param cycles 推理循环次数（本实现中未使用）
 * @return 返回推理结果字符串
 */
- (NSString *)inferImage:(UIImage *)image cycles:(NSInteger)cycles {
    // 加锁保护，确保线程安全
    std::unique_lock<std::mutex> _l(_mutex);
    
    // 获取输入图像的宽高
    int w = image.size.width;
    int h = image.size.height;
    
    // 分配RGBA图像缓冲区，每个像素4个通道
    unsigned char *rgba = (unsigned char *)calloc(w * h * 4, sizeof(unsigned char));
    
    // 将UIImage转换为RGBA格式
    {
        CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
        CGContextRef contextRef = CGBitmapContextCreate(rgba, w, h, 8, w * 4, colorSpace,
                                                      kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
        CGContextDrawImage(contextRef, CGRectMake(0, 0, w, h), image.CGImage);
        CGContextRelease(contextRef);
    }

    // 设置图像预处理参数
    const float means[3] = {103.94f, 116.78f, 123.68f};    // 均值
    const float normals[3] = {0.017f, 0.017f, 0.017f};     // 归一化参数
    
    // 创建图像预处理器，设置输入格式为RGBA，输出格式为BGR
    auto pretreat = std::shared_ptr<MNN::CV::ImageProcess>(
        MNN::CV::ImageProcess::create(MNN::CV::RGBA, MNN::CV::BGR, means, 3, normals, 3));
    
    
    MNN::CV::Matrix matrix;
    matrix.postScale((w - 1) / 223.0, (h - 1) / 223.0);
    pretreat->setMatrix(matrix);

    // 获取模型的输入张量
    auto input = _net->getSessionInput(_session, nullptr);
    
    // 执行图像预处理并将结果写入输入张量
    pretreat->convert(rgba, w, h, 0, input);
    
    // 释放临时分配的图像缓冲区
    free(rgba);

    // 调用父类的推理方法执行实际的模型推理
    return [super inferNoLock:0];
} 
- (NSString *)inferBuffer:(CMSampleBufferRef)sampleBuffer {
    CVPixelBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    std::unique_lock<std::mutex> _l(_mutex);

    // GPU
    if (_type == MNN_FORWARD_METAL) {
        size_t width = CVPixelBufferGetWidth(pixelBuffer);
        size_t height = CVPixelBufferGetHeight(pixelBuffer);
        MTLPixelFormat pixelFormat = MTLPixelFormatBGRA8Unorm;

        CVMetalTextureRef texture = NULL;
        CVReturn status = CVMetalTextureCacheCreateTextureFromImage(NULL, _cache->_textureCache, pixelBuffer, NULL, pixelFormat, width, height, 0, &texture);
        id<MTLTexture> inputTexture = CVMetalTextureGetTexture(texture);
        CVBufferRelease(texture);
        PretreatInfo pretreat;
        // TODO: Only copy it once
        pretreat.outputSize[0] = 224;
        pretreat.outputSize[1] = 224;
        pretreat.mean[0] = 103.94f;
        pretreat.mean[1] = 116.78f;
        pretreat.mean[2] = 123.68f;
        pretreat.mean[3] = 0.0f;
        pretreat.normal[0] = 0.017f;
        pretreat.normal[1] = 0.017f;
        pretreat.normal[2] = 0.017f;
        pretreat.normal[3] = 0.0f;
        ::memcpy([_cache->_constant contents], &pretreat, sizeof(PretreatInfo));
        auto cmd = [_cache->_queue commandBuffer];
        auto enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:_cache->_pretreat];
        [enc setTexture:inputTexture atIndex:0];
        [enc setBuffer:_cache->_constant offset:0 atIndex:1];
        MNNMetalTensorContent sharedContent;
        _input->getDeviceInfo(&sharedContent, MNN_FORWARD_METAL);
        // For Metal Context to write, don't need finish, just use flush
        _input->wait(MNN::Tensor::MAP_TENSOR_WRITE, false);
        [enc setBuffer:sharedContent.buffer offset:sharedContent.offset atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(28, 28, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];        
        return [super inferNoLock:0];
    }

    // CPU
    int w                        = (int)CVPixelBufferGetWidth(pixelBuffer);
    int h                        = (int)CVPixelBufferGetHeight(pixelBuffer);
    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    unsigned char *bgra = (unsigned char *)CVPixelBufferGetBaseAddress(pixelBuffer);

    const float means[3]   = {103.94f, 116.78f, 123.68f};
    const float normals[3] = {0.017f, 0.017f, 0.017f};
    auto pretreat          = std::shared_ptr<MNN::CV::ImageProcess>(
    MNN::CV::ImageProcess::create(MNN::CV::BGRA, MNN::CV::BGR, means, 3, normals, 3));
    MNN::CV::Matrix matrix;
    matrix.postScale((w - 1) / 223.0, (h - 1) / 223.0);
    pretreat->setMatrix(matrix);

    auto input = _net->getSessionInput(_session, nullptr);
    pretreat->convert(bgra, w, h, 0, input);

    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    return [super inferNoLock:0];
}

@end

#pragma mark -
@interface SqueezeNetV1_1 : Model
@end

@implementation SqueezeNetV1_1

- (instancetype)init {
    self = [super init];
    if (self) {
        NSString *labels  = [[NSBundle mainBundle] pathForResource:@"squeezenet" ofType:@"txt"];
        NSString *lines   = [NSString stringWithContentsOfFile:labels encoding:NSUTF8StringEncoding error:nil];
        self.labels       = [lines componentsSeparatedByString:@"\n"];
        self.defaultImage = [UIImage imageNamed:@"squeezenet.jpg"];

        NSString *model = [[NSBundle mainBundle] pathForResource:@"squeezenet_v1.1.caffe" ofType:@"mnn"];
        _net            = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model.UTF8String));
    }
    return self;
}

- (NSString *)inferImage:(UIImage *)image cycles:(NSInteger)cycles {
    std::unique_lock<std::mutex> _l(_mutex);
    int w               = image.size.width;
    int h               = image.size.height;
    unsigned char *rgba = (unsigned char *)calloc(w * h * 4, sizeof(unsigned char));
    {
        CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
        CGContextRef contextRef    = CGBitmapContextCreate(rgba, w, h, 8, w * 4, colorSpace,
                                                        kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
        CGContextDrawImage(contextRef, CGRectMake(0, 0, w, h), image.CGImage);
        CGContextRelease(contextRef);
    }

    const float means[3] = {104.f, 117.f, 123.f};
    MNN::CV::ImageProcess::Config process;
    ::memcpy(process.mean, means, sizeof(means));
    process.sourceFormat = MNN::CV::RGBA;
    process.destFormat   = MNN::CV::BGR;

    std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(process));
    MNN::CV::Matrix matrix;
    matrix.postScale((w - 1) / 226.f, (h - 1) / 226.f);
    pretreat->setMatrix(matrix);

    auto input = _net->getSessionInput(_session, nullptr);
    pretreat->convert(rgba, w, h, 0, input);
    free(rgba);

    return [super inferNoLock:0];
}

- (NSString *)inferBuffer:(CMSampleBufferRef)sampleBuffer {
    std::unique_lock<std::mutex> _l(_mutex);
    CVPixelBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    int w                        = (int)CVPixelBufferGetWidth(pixelBuffer);
    int h                        = (int)CVPixelBufferGetHeight(pixelBuffer);

    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    unsigned char *bgra = (unsigned char *)CVPixelBufferGetBaseAddress(pixelBuffer);

    const float means[3] = {104.f, 117.f, 123.f};
    MNN::CV::ImageProcess::Config process;
    ::memcpy(process.mean, means, sizeof(means));
    process.sourceFormat = MNN::CV::BGRA;
    process.destFormat   = MNN::CV::BGR;

    std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(process));
    MNN::CV::Matrix matrix;
    matrix.postScale((w - 1) / 226.f, (h - 1) / 226.f);
    pretreat->setMatrix(matrix);

    auto input = _net->getSessionInput(_session, nullptr);
    pretreat->convert(bgra, w, h, 0, input);

    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    return [super inferNoLock:0];
}
@end

@interface ViewController ()<AVCaptureVideoDataOutputSampleBufferDelegate>
@property (assign, nonatomic) MNNForwardType forwardType;
@property (assign, nonatomic) int threadCount;

@property (strong, nonatomic) Model *mobileNetV2;
@property (strong, nonatomic) Model *squeezeNetV1_1;
@property (strong, nonatomic) Model *currentModel;

@property (strong, nonatomic) AVCaptureSession *session;
@property (strong, nonatomic) IBOutlet UIImageView *imageView;
@property (strong, nonatomic) IBOutlet UILabel *resultLabel;
@property (strong, nonatomic) IBOutlet UIBarButtonItem *modelItem;
@property (strong, nonatomic) IBOutlet UIBarButtonItem *forwardItem;
@property (strong, nonatomic) IBOutlet UIBarButtonItem *threadItem;
@property (strong, nonatomic) IBOutlet UIBarButtonItem *runItem;
@property (strong, nonatomic) IBOutlet UIBarButtonItem *benchmarkItem;
@property (strong, nonatomic) IBOutlet UIBarButtonItem *cameraItem;
//@property (weak, nonatomic) IBOutlet UIButton *customItem;
@property (strong, nonatomic) IBOutlet UIBarButtonItem *customItem;  // 新增自定义按钮

@end

@implementation ViewController

- (void)awakeFromNib {
    [super awakeFromNib];

    // 初始化基本设置
    self.forwardType    = MNN_FORWARD_CPU;
    self.threadCount    = 4;
    self.mobileNetV2    = [MobileNetV2 new];
    self.squeezeNetV1_1 = [SqueezeNetV1_1 new];
    self.currentModel   = self.mobileNetV2;
    
    // 设置自定义按钮标题
    self.customItem.title = @"自定义操作";

    AVCaptureSession *session        = [[AVCaptureSession alloc] init];
    session.sessionPreset            = AVCaptureSessionPreset1280x720;
    AVCaptureDevice *device          = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    AVCaptureDeviceInput *input      = [[AVCaptureDeviceInput alloc] initWithDevice:device error:NULL];
    AVCaptureVideoDataOutput *output = [[AVCaptureVideoDataOutput alloc] init];
    [output setSampleBufferDelegate:self queue:dispatch_queue_create("video_infer", 0)];
    output.videoSettings = @{(id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA)};

    if ([session canAddInput:input]) {
        [session addInput:input];
    }
    if ([session canAddOutput:output]) {
        [session addOutput:output];
    }
    [session commitConfiguration];

    self.session = session;
}

- (void)viewDidAppear:(BOOL)animated {
    [super viewDidAppear:animated];
    [self refresh];
}

- (void)refresh {
    [_currentModel setType:_forwardType threads:_threadCount];
    [self run];
}

- (IBAction)toggleInput {
    if (_session.running) {
        [self usePhotoInput];
        [self run];
    } else {
        [self useCameraInput];
    }
}

- (void)useCameraInput {
    [_session startRunning];
    self.navigationItem.leftBarButtonItem.title = @"Photo";
    self.runItem.enabled                        = NO;
    self.benchmarkItem.enabled                  = NO;
}

- (void)usePhotoInput {
    [_session stopRunning];
    _imageView.image                            = _currentModel.defaultImage;
    self.navigationItem.leftBarButtonItem.title = @"Camera";
    self.runItem.enabled                        = YES;
    self.benchmarkItem.enabled                  = YES;
}

- (IBAction)toggleModel {
    __weak typeof(self) weakify = self;
    UIAlertController *alert    = [UIAlertController alertControllerWithTitle:@"选择模型"
                                                                   message:nil
                                                            preferredStyle:UIAlertControllerStyleActionSheet];
    [alert addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    [alert addAction:[UIAlertAction actionWithTitle:@"MobileNet V2"
                                              style:UIAlertActionStyleDefault
                                            handler:^(UIAlertAction *action) {
                                                __strong typeof(weakify) self = weakify;
                                                self.modelItem.title          = action.title;
                                                self.currentModel             = self.mobileNetV2;
                                                if (!self.session.running) {
                                                    self.imageView.image = self.currentModel.defaultImage;
                                                }
                                                [self refresh];
                                            }]];
    [alert addAction:[UIAlertAction actionWithTitle:@"SqueezeNet V1.1"
                                              style:UIAlertActionStyleDefault
                                            handler:^(UIAlertAction *action) {
                                                __strong typeof(weakify) self = weakify;
                                                self.modelItem.title          = action.title;
                                                self.currentModel             = self.squeezeNetV1_1;
                                                if (!self.session.running) {
                                                    self.imageView.image = self.currentModel.defaultImage;
                                                }
                                                [self refresh];
                                            }]];
    [self presentViewController:alert animated:YES completion:nil];
}

- (IBAction)toggleMode {
    __weak typeof(self) weakify = self;
    UIAlertController *alert    = [UIAlertController alertControllerWithTitle:@"运行模式"
                                                                   message:nil
                                                            preferredStyle:UIAlertControllerStyleActionSheet];
    [alert addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    [alert addAction:[UIAlertAction actionWithTitle:@"CPU"
                                              style:UIAlertActionStyleDefault
                                            handler:^(UIAlertAction *action) {
                                                __strong typeof(weakify) self = weakify;
                                                self.forwardItem.title        = action.title;
                                                self.forwardType              = MNN_FORWARD_CPU;
                                                [self refresh];
                                            }]];
    [alert addAction:[UIAlertAction actionWithTitle:@"Metal"
                                              style:UIAlertActionStyleDefault
                                            handler:^(UIAlertAction *action) {
                                                __strong typeof(weakify) self = weakify;
                                                self.forwardItem.title        = action.title;
                                                self.forwardType              = MNN_FORWARD_METAL;
                                                [self refresh];
                                            }]];
    [self presentViewController:alert animated:YES completion:nil];
}

- (IBAction)toggleThreads {
    __weak typeof(self) weakify       = self;
    void (^onToggle)(UIAlertAction *) = ^(UIAlertAction *action) {
        __strong typeof(weakify) self = weakify;
        self.threadItem.title         = [NSString stringWithFormat:@"%@", action.title];
        self.threadCount              = action.title.intValue;
        [self refresh];
    };
    UIAlertController *alert = [UIAlertController alertControllerWithTitle:@"Thread Count"
                                                                   message:nil
                                                            preferredStyle:UIAlertControllerStyleActionSheet];
    [alert addAction:[UIAlertAction actionWithTitle:@"取消" style:UIAlertActionStyleCancel handler:nil]];
    [alert addAction:[UIAlertAction actionWithTitle:@"1" style:UIAlertActionStyleDefault handler:onToggle]];
    [alert addAction:[UIAlertAction actionWithTitle:@"2" style:UIAlertActionStyleDefault handler:onToggle]];
    [alert addAction:[UIAlertAction actionWithTitle:@"4" style:UIAlertActionStyleDefault handler:onToggle]];
    [alert addAction:[UIAlertAction actionWithTitle:@"8" style:UIAlertActionStyleDefault handler:onToggle]];
    [alert addAction:[UIAlertAction actionWithTitle:@"10" style:UIAlertActionStyleDefault handler:onToggle]];
    [self presentViewController:alert animated:YES completion:nil];
}

- (IBAction)run {
    if (!_session.running) {
        self.resultLabel.text = [_currentModel inferImage:_imageView.image cycles:1];
    }
}

-(void) linear_cpu {
     if (!_session.running) {
        // 禁用所有按钮，防止操作冲突
        
        //self.customItem.enabled    = NO;//
        
        NSLog(@"开始自定义操作...");
        // 加载MNN模型
        NSString* modelPath = [[NSBundle mainBundle] pathForResource:@"linear_test_model" ofType:@"mnn"];
        
        
        auto interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile([modelPath UTF8String]));
        
            
        if (!interpreter) {
            NSLog(@"Failed to create interpreter");
            return;
        }
        //auto session = interpreter->createSession(nullptr); // nullptr 表示使用默认配置
        MNN::ScheduleConfig scheduleConfig;
        scheduleConfig.type = MNN_FORWARD_CPU;
        auto session=interpreter->createSession(scheduleConfig);
        // 获取输入张量
            auto inputTensor = interpreter->getSessionInput(session, NULL);
            auto inputShape = inputTensor->shape(); // 获取形状
            
            int size = 1;
            for (int i = 0; i < inputShape.size(); ++i) {
                size *= inputShape[i];
                NSLog(@"inputShape[%d]: %d", i, inputShape[i]);
            }
            NSLog(@"size: %d", size);
            
            float inputData[size];
        for (int i = 0; i < size; ++i) {
            inputData[i] = ((float)rand() / RAND_MAX); // 随机初始化
                NSLog(@"inputData[%d]: %f", i, inputData[i]);
        }
            //inputData[0] = 1.0f;
        
          
        // 创建 Host Tensor 并拷贝数据
        MNN::Tensor* hostTensor = MNN::Tensor::create(inputShape, halide_type_of<float>());
        memcpy(hostTensor->host<float>(), inputData, sizeof(float) * size);

        //// 创建 Device Tensor 并从 Host 拷贝过去
        MNN::Tensor* deviceTensor = MNN::Tensor::createDevice(inputShape, halide_type_of<float>());
        deviceTensor->copyFromHostTensor(hostTensor);

        // 设置输入
        //auto inputTensor = interpreter->getSessionInput(session, nullptr);
        inputTensor->copyFromHostTensor(hostTensor);

        // 运行推理
        interpreter->runSession(session);

        // 获取输出
        auto outputTensor = interpreter->getSessionOutput(session, nullptr);
        //auto outputHostTensor = MNN::Tensor::create(outputTensor, outputTensor->getDimensionType());
        //outputTensor->copyToHostTensor(outputHostTensor);

        // 输出结果
        float *outputData=outputTensor->host<float>();
        //float* outputData = outputHostTensor->host<float>();
        NSLog(@"原始结果 y=2*x+0.01: %f", 2*inputData[0] + 0.01);
        NSLog(@"预测输出a: %f", outputData[0]);

        // 清理资源
        MNN::Tensor::destroy(hostTensor);
        MNN::Tensor::destroy(deviceTensor);
        //MNN::Tensor::destroy(outputHostTensor);
        

            
    }
}

-(void) linear_gpu {

     if (!_session.running) {
        // 禁用所有按钮，防止操作冲突
        
        //self.customItem.enabled    = NO;//
            // 加锁保护，确保线程安全
        
         std::shared_ptr<GpuCache> _cache;
         // 如果GPU缓存未初始化，创建新的GPU资源缓存
         if (nullptr == _cache) {
             _cache.reset(new GpuCache);
         }
         
        NSLog(@"开始 GPU 推理...");
        // 加载MNN模型
        NSString* modelPath = [[NSBundle mainBundle] pathForResource:@"linear_test_model" ofType:@"mnn"];
        
        
        auto _net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile([modelPath UTF8String]));
        
            
        if (!_net) {
            NSLog(@"Failed to create interpreter");
            return;
        }
        //auto session = interpreter->createSession(nullptr); // nullptr 表示使用默认配置
        MNN::ScheduleConfig scheduleConfig;
        scheduleConfig.type = MNN_FORWARD_METAL;
         MNN::Session *_session;
         if (scheduleConfig.type == MNN_FORWARD_METAL) {
             // Metal GPU模式下的特殊配置
             MNN::BackendConfig bnConfig;
             MNNMetalSharedContext context;
             // 设置Metal上下文，复用已创建的设备和命令队列
             context.device = _cache->_device;
             context.queue = _cache->_queue;
             bnConfig.sharedContext = &context;
             scheduleConfig.backendConfig = &bnConfig;
             _session = _net->createSession(scheduleConfig);
         }
        
        //auto session=interpreter->createSession(scheduleConfig);
        // 获取输入张量
            auto inputTensor = _net->getSessionInput(_session, NULL);
            auto inputShape = inputTensor->shape(); // 获取形状
            
            int size = 1;
            for (int i = 0; i < inputShape.size(); ++i) {
                size *= inputShape[i];
                NSLog(@"inputShape[%d]: %d", i, inputShape[i]);
            }
            NSLog(@"size: %d", size);
            
            float inputData[size];
        for (int i = 0; i < size; ++i) {
            inputData[i] = ((float)rand() / RAND_MAX); // 随机初始化
                NSLog(@"inputData[%d]: %f", i, inputData[i]);
        }
            //inputData[0] = 1.0f;
        
          
        // 创建 Host Tensor 并拷贝数据
        MNN::Tensor* hostTensor = MNN::Tensor::create(inputShape, halide_type_of<float>());
        memcpy(hostTensor->host<float>(), inputData, sizeof(float) * size);

        //// 创建 Device Tensor 并从 Host 拷贝过去


        // 设置输入
        //auto inputTensor = interpreter->getSessionInput(session, nullptr);
        inputTensor->copyFromHostTensor(hostTensor);

        // 运行推理
         _net->runSession(_session);

        // 获取输出
        auto outputTensor = _net->getSessionOutput(_session, nullptr);
        MNN::Tensor  outputHostTensor (outputTensor);
        outputTensor->copyToHostTensor(&outputHostTensor);

        // 输出结果
        float *outputData=outputHostTensor.host<float>();
        //float* outputData = outputHostTensor->host<float>();
        NSLog(@"原始结果 y=2*x+0.01: %f", 2*inputData[0] + 0.01);
        NSLog(@"预测输出a: %f", outputData[0]);

        // 清理资源
         
         
        // 将在GPU上返回的数据再次送给模型进行推理
         //inputTensor(outputHostTensor);
        MNN::Tensor::destroy(hostTensor);
        
        //MNN::Tensor::destroy(outputHostTensor);
        

            
    }
}


- (IBAction)customAction {
    [self linear_gpu];
}
// 自定义按钮动作处理方法
- (IBAction)customAction2 {
    if (!_session.running) {
        // 禁用所有按钮，防止操作冲突
        
        //self.customItem.enabled    = NO;//
        
        NSLog(@"开始自定义操作...");
        // 加载MNN模型
        NSString* modelPath = [[NSBundle mainBundle] pathForResource:@"linear_test_model" ofType:@"mnn"];
        
        
        auto interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile([modelPath UTF8String]));
        
            
        if (!interpreter) {
            NSLog(@"Failed to create interpreter");
            return;
        }
        //auto session = interpreter->createSession(nullptr); // nullptr 表示使用默认配置
        MNN::ScheduleConfig scheduleConfig;
        scheduleConfig.type = MNN_FORWARD_CPU;
        auto session=interpreter->createSession(scheduleConfig);
        // 获取输入张量
            auto inputTensor = interpreter->getSessionInput(session, NULL);
            auto inputShape = inputTensor->shape(); // 获取形状
            
            int size = 1;
            for (int i = 0; i < inputShape.size(); ++i) {
                size *= inputShape[i];
                NSLog(@"inputShape[%d]: %d", i, inputShape[i]);
            }
            NSLog(@"size: %d", size);
            
            float inputData[size];
        for (int i = 0; i < size; ++i) {
            inputData[i] = ((float)rand() / RAND_MAX); // 随机初始化
                NSLog(@"inputData[%d]: %f", i, inputData[i]);
        }
            //inputData[0] = 1.0f;
        
          
        // 创建 Host Tensor 并拷贝数据
        MNN::Tensor* hostTensor = MNN::Tensor::create(inputShape, halide_type_of<float>());
        memcpy(hostTensor->host<float>(), inputData, sizeof(float) * size);

        //// 创建 Device Tensor 并从 Host 拷贝过去
        MNN::Tensor* deviceTensor = MNN::Tensor::createDevice(inputShape, halide_type_of<float>());
        deviceTensor->copyFromHostTensor(hostTensor);

        // 设置输入
        //auto inputTensor = interpreter->getSessionInput(session, nullptr);
        inputTensor->copyFromHostTensor(hostTensor);

        // 运行推理
        interpreter->runSession(session);

        // 获取输出
        auto outputTensor = interpreter->getSessionOutput(session, nullptr);
        //auto outputHostTensor = MNN::Tensor::create(outputTensor, outputTensor->getDimensionType());
        //outputTensor->copyToHostTensor(outputHostTensor);

        // 输出结果
        float *outputData=outputTensor->host<float>();
        //float* outputData = outputHostTensor->host<float>();
        NSLog(@"原始结果 y=2*x+0.01: %f", 2*inputData[0] + 0.01);
        NSLog(@"预测输出: %f", outputData[0]);

        // 清理资源
        MNN::Tensor::destroy(hostTensor);
        MNN::Tensor::destroy(deviceTensor);
        //MNN::Tensor::destroy(outputHostTensor);
        

            
    }
}

/**
 * 执行模型性能基准测试
 * 在后台线程运行100次推理，计算平均执行时间
 */
- (IBAction)benchmark {
    // 检查相机是否在运行，只有在非相机模式下才能执行基准测试
    if (!_session.running) {
        // 禁用所有UI控件，防止测试过程中的用户交互
        self.cameraItem.enabled    = NO;    // 禁用相机切换按钮
        self.runItem.enabled       = NO;    // 禁用运行按钮
        self.benchmarkItem.enabled = NO;    // 禁用基准测试按钮
        self.modelItem.enabled     = NO;    // 禁用模型选择按钮
        self.forwardItem.enabled   = NO;    // 禁用推理后端选择按钮
        self.threadItem.enabled    = NO;    // 禁用线程数选择按钮
        
        // 在全局后台队列中异步执行基准测试
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            // 执行100次推理并获取性能统计结果
            NSString *str = [self->_currentModel benchmark:100];
            
            // 在主线程更新UI
            dispatch_async(dispatch_get_main_queue(), ^{
                // 显示测试结果
                self.resultLabel.text      = str;
                
                // 重新启用所有UI控件
                self.cameraItem.enabled    = YES;
                self.runItem.enabled       = YES;
                self.benchmarkItem.enabled = YES;
                self.modelItem.enabled     = YES;
                self.forwardItem.enabled   = YES;
                self.threadItem.enabled    = YES;
            });
        });
    }
}

#pragma mark AVCaptureAudioDataOutputSampleBufferDelegate
- (void)captureOutput:(AVCaptureOutput *)output
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
           fromConnection:(AVCaptureConnection *)connection {
    CIImage *ci        = [[CIImage alloc] initWithCVPixelBuffer:CMSampleBufferGetImageBuffer(sampleBuffer)];
    CIContext *context = [[CIContext alloc] init];
    CGImageRef cg      = [context createCGImage:ci fromRect:ci.extent];

    UIImageOrientation orientaion;
    switch (connection.videoOrientation) {
        case AVCaptureVideoOrientationPortrait:
            orientaion = UIImageOrientationUp;
            break;
        case AVCaptureVideoOrientationPortraitUpsideDown:
            orientaion = UIImageOrientationDown;
            break;
        case AVCaptureVideoOrientationLandscapeRight:
            orientaion = UIImageOrientationRight;
            break;
        case AVCaptureVideoOrientationLandscapeLeft:
            orientaion = UIImageOrientationLeft;
            break;
        default:
            break;
    }

    UIImage *image = [UIImage imageWithCGImage:cg scale:1.f orientation:orientaion];
    CGImageRelease(cg);
    NSString *result = [_currentModel inferBuffer:sampleBuffer];

    dispatch_async(dispatch_get_main_queue(), ^{
        if (self.session.running) {
            self.imageView.image  = image;
            self.resultLabel.text = result;
        }
    });
}

@end
